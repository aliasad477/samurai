// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#include <cmath>
#include <iostream>
#include <samurai/io/hdf5.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>
#include <samurai/schemes/fv.hpp>

#include "ts_assembly_and_solve.hpp"

int main(int argc, char* argv[])
{
    auto& app = samurai::initialize("Differentially heated cavity", argc, argv);

    constexpr std::size_t dim = 2;

    //----------------//
    //   Parameters   //
    //----------------//

    double Ra = 1e5;  // Rayleigh number
    double Pr = 0.71; // Prandtl number (air)
    double diff_constant = 1.0;

    double Tf = 10; // Final time
    double dt = 1e-1;

    std::size_t level_min = 2;
    std::size_t level_max = 7;

    std::size_t nfiles   = 0;
    fs::path path        = fs::current_path();
    std::string filename = "dhc";

    app.add_option("--Ra", Ra, "Rayleigh number")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Pr", Pr, "Prandtl number")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Tf", Tf, "Final time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--dt", dt, "Time step")->capture_default_str()->group("Simulation parameters");
    app.add_option("--level-min", level_min, "Minimum mesh level")->capture_default_str()->group("Simulation parameters");
    app.add_option("--level-max", level_max, "Maximum mesh level")->capture_default_str()->group("Simulation parameters");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Output");
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Output");
    app.add_option("--nfiles", nfiles, "Number of output files")->capture_default_str()->group("Output");
    SAMURAI_PARSE(argc, argv);

    if (!fs::exists(path))
    {
        fs::create_directory(path);
    }

    std::cout << "Differentially heated cavity (Ra = " << Ra << ")" << std::endl;

    // Dimensionless incompressible Navier-Stokes equations (see, e.g., https://www.mdpi.com/2673-3951/6/3/66 eq. (19)-(22)):
    //
    //              ∂V/∂t -   Pr*ΔV + V·∇V + ∇P - Ra*Pr*T*e_y = 0
    //                                                   ∇·V = 0
    //              ∂T/∂t - ΔT + V·∇T              = 0
    //
    // where V = velocity,
    //       P = pressure,
    //       T = temperature,
    // and Ra  = Rayleigh number,
    //     Pr  = Prandtl number,
    //     e_y = unit vector in the y direction

    //-----------------//
    // Field creations //
    //-----------------//

    // Mesh creation
    auto box    = samurai::Box<double, dim>({0, 0}, {1, 1});
    auto config = samurai::mesh_config<dim>().min_level(level_min).max_level(level_max).max_stencil_size(2);
    auto mesh   = samurai::mra::make_mesh(box, config);

    // Fields for the Navier-Stokes equations
    auto velocity        = samurai::make_vector_field<double, dim>("velocity", mesh);
    auto pressure        = samurai::make_scalar_field<double>("pressure", mesh);
    auto temperature     = samurai::make_scalar_field<double>("temperature", mesh);

    // Fields for the right-hand side of the system
    auto zero_pressure = samurai::make_scalar_field<double>("zero_pressure", mesh);

    // Fields for the null space of the system (constant pressure)
    auto constant_pressure = samurai::make_scalar_field<double>("constant_pressure", mesh, 1.);
    auto zero_velocity     = samurai::make_vector_field<double, dim>("zero_velocity", mesh, 0.);
    auto zero_temperature  = samurai::make_scalar_field<double>("zero_temperature", mesh, 0.);

    // using VelocityField    = decltype(velocity);
    // using PressureField    = decltype(pressure);
    // using TemperatureField = decltype(temperature);

    //---------------------//
    // Boundary conditions //
    //---------------------//

    samurai::DirectionVector<dim> left   = {-1, 0};
    samurai::DirectionVector<dim> right  = {1, 0};
    samurai::DirectionVector<dim> bottom = {0, -1};
    samurai::DirectionVector<dim> top    = {0, 1};

    // No-slip velocity on all walls
    samurai::make_bc<samurai::Dirichlet<1>>(velocity, 0., 0.);

    // Temperature boundary conditions
    samurai::make_bc<samurai::Dirichlet<1>>(temperature, 1.)->on(left);      // Hot wall (T = 1)
    samurai::make_bc<samurai::Dirichlet<1>>(temperature, 0.)->on(right);     // Cold wall (T = 0)
    samurai::make_bc<samurai::Neumann<1>>(temperature, 0.)->on(top, bottom); // Adiabatic walls (∂T/∂n = 0)

    samurai::make_bc<samurai::Neumann<1>>(pressure, 0.);

    // Multi-resolution: the mesh will be adapted according to the velocity
    auto MRadaptation = samurai::make_MRAdapt(velocity);
    auto mra_config   = samurai::mra_config().epsilon(1e-3).regularity(2);

    //--------------------//
    // Initial conditions //
    //--------------------//

    velocity.fill(0);
    pressure.fill(0);
    zero_pressure.fill(0);
    // Initial temperature: linear profile from cold (right) to hot (left)
    samurai::for_each_cell(mesh,
                           [&](auto& cell)
                           {
                               double x          = cell.center(0);
                               temperature[cell] = 1. - x; // Linear profile: T = 1 at x=0, T = 0 at x=1
                           });

    
    double dt_save    = nfiles == 0 ? dt : Tf / static_cast<double>(nfiles);
    std::size_t nsave = 0, nt = 0;

    if (nfiles != 1)
    {
        samurai::save(path, fmt::format("dhc_velocity_ite_{}", nsave), mesh, velocity);
        samurai::save(path, fmt::format("dhc_temperature_ite_{}", nsave), mesh, temperature);
        samurai::save(path, fmt::format("dhc_pressure_ite_{}", nsave), mesh, pressure);
        nsave++;
    }

    //----------------//
    // Time iteration //
    //----------------//
    
    // Resgister lms_esdirk5 method for TSDIRK (if used)
    // Register_LMS_ESDIRK5();

    double t = 0;
    while (t < Tf)
    {
        // Move to next timestep
        t += dt;
        if (t > Tf)
        {
            dt += Tf - t;
            if (dt < 1e-12)
            {
                break;
            }
            t = Tf;
        }
        std::cout << fmt::format("iteration {}: t = {:.2f}, dt = {}\n", nt++, t, dt);

        // Mesh adaptation for Navier-Stokes
        if (mesh.min_level() != mesh.max_level())
        {
            // Current pressure and temperature values must be conserved on the new mesh
            // to be used in the right-hand side of the non-linear system and as initial guess for the Newton method.
            MRadaptation(mra_config, temperature, pressure);

            velocity.resize();
            pressure.resize();
            temperature.resize();

            zero_pressure.resize();
            zero_pressure.fill(0);

        }

        configure_and_solve_fluid_ts(velocity,
                                     pressure,
                                     temperature,
                                     diff_constant,
                                     Pr,
                                     Ra,
                                     dt,
                                     t);

        // Remove the average pressure to avoid drift
        double avg_pressure = 0.0;
        double sum_volumes  = 0.0;
        samurai::for_each_cell(mesh,
                               [&](const auto& cell)
                               {
                                   double volume = std::pow(cell.length, dim);
                                   avg_pressure += volume * pressure[cell];
                                   sum_volumes += volume;
                               });
        avg_pressure /= sum_volumes;
        pressure = pressure - avg_pressure;

        // Save the results
        if (t >= static_cast<double>(nsave) * dt_save || t == Tf)
        {
            if (nfiles != 1)
            {
                samurai::save(path, fmt::format("dhc_velocity_ite_{}", nsave), mesh, velocity);
                samurai::save(path, fmt::format("dhc_temperature_ite_{}", nsave), mesh, temperature);
                samurai::save(path, fmt::format("dhc_pressure_ite_{}", nsave), mesh, pressure);
            }
            nsave++;
        }
    } // end time loop
    
    // Ensure saving only at the final time for the case nfiles=1
    if (nfiles == 1)
    {
        if (level_max == level_min){
            samurai::save(path, fmt::format("{}_velocity", filename), mesh, velocity);
            samurai::save(path, fmt::format("{}_temperature", filename), mesh, temperature);
            samurai::save(path, fmt::format("{}_pressure", filename), mesh, pressure);
        } else{
            auto config_fine = samurai::mesh_config<dim>().min_level(level_max).max_level(level_max).max_stencil_size(2);
            auto mesh_fine   = samurai::mra::make_mesh(box, config_fine);
            
            auto temperature_fine   = samurai::make_scalar_field<double>("temperature", mesh_fine);
            auto pressure_fine      = samurai::make_scalar_field<double>("pressure", mesh_fine);
            auto velocity_fine      = samurai::make_vector_field<double, dim>("velocity", mesh_fine);
            
            temperature_fine.fill(0.);
            pressure_fine.fill(0.);
            velocity_fine.fill(0.);

            samurai::transfer(temperature, temperature_fine);
            samurai::transfer(pressure, pressure_fine);
            samurai::transfer(velocity, velocity_fine);
                                
            samurai::save(path, fmt::format("{}_velocity", filename), mesh_fine, velocity_fine);
            samurai::save(path, fmt::format("{}_temperature", filename), mesh_fine, temperature_fine);
            samurai::save(path, fmt::format("{}_pressure", filename), mesh_fine, pressure_fine);

        }
    }

    samurai::finalize();
    return 0;
}
