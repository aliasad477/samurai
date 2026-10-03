#pragma once

#include <samurai/field.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/schemes/fv.hpp>
#include <petsc.h>

template <class Field, typename value_t>
auto make_scaled_identity(value_t &scalar)
{
    static constexpr std::size_t n_comp = Field::n_comp;
    using field_value_type              = typename Field::value_type;

    using cfg = samurai::LocalCellSchemeConfig<samurai::SchemeType::LinearHomogeneous, Field, Field>;

    auto scaled_identity = samurai::make_cell_based_scheme<cfg>("ScaledIdentity");

    scaled_identity.coefficients_func() = [&](samurai::StencilCoeffs<cfg>& sc, double)
    {
        sc = scalar * samurai::eye<field_value_type, n_comp, n_comp, Field::is_scalar>();
    };
    scaled_identity.is_symmetric(true);
    scaled_identity.is_spd(true);
    return scaled_identity;
}

template<class VelocityField, class ScalarField>
struct FluidAssembly
{
    static constexpr std::size_t dim = VelocityField::dim;

    template<samurai::petsc::BlockAssemblyType bMonolithic>
    inline auto get_assembly(VelocityField& velocity,
                             ScalarField&   pressure,
                             ScalarField&   temperature,
                             double& thermal_diffusivity,
                             double& Pr,
                             double& Ra,
                             double a) const
    {

        // coeffs to operators
        samurai::DiffCoeff<dim> T_diff_coeff;
        T_diff_coeff.fill( thermal_diffusivity );
        samurai::DiffCoeff<dim> u_diff_coeff;
        u_diff_coeff.fill( Pr );
        double bu_fac = Pr * Ra;

        auto diff_V = samurai::make_diffusion_order2<VelocityField>(u_diff_coeff);       // diff: V ---> PrΔV
        auto conv_V = samurai::make_convection_smooth_rusanov_incompressible<VelocityField>(); // conv: V ---> V·∇V
        auto div_V  = samurai::make_divergence_order2<VelocityField>();                        // div:  V ---> ∇·V
        auto a_V    = make_scaled_identity<VelocityField>(a);

        // used for the assembly of the Jacobian matrix: it fills the block ∂(conv_T)/∂V
        auto conv_dual = samurai::make_dual_convection_smooth_rusanov_incompressible<ScalarField, VelocityField>(temperature);
        // conv_V.set_name("Convection_V");

        auto grad_P = samurai::make_gradient_order2<ScalarField>(); // grad: P ---> ∇P
        auto zero_P = samurai::make_zero_operator<ScalarField>();   // zero: P ---> 0

        auto diff_T = samurai::make_diffusion_order2<ScalarField>(T_diff_coeff);              // diff: T ---> ΔT
        auto conv_T = samurai::make_convection_smooth_rusanov_incompressible<ScalarField>(velocity); // conv: T ---> V·∇T
        auto buoy_T = samurai::make_buoyancy<VelocityField, ScalarField>(bu_fac); // buoy: T ---> -(RaPr)T*e_y (acts only in y-direction)// auto a_T    = samurai::make_identity<ScalarField>();

        auto a_T    = make_scaled_identity<ScalarField>(a);
        // conv_T.set_name("Convection_T");

        auto op_00 = a_V + (diff_V + conv_V);
        auto op_01 = grad_P;
        auto op_02 = buoy_T;

        auto op_10 = -div_V;
        auto op_11 = zero_P;

        auto op_20 = conv_dual;
        auto op_22 = a_T + (diff_T + conv_T);

        // Define the block operator
        auto system_fluid = samurai::make_block_operator<3, 3> (op_00,      op_01,       op_02,
                                                                op_10,      op_11,       0,
                                                                op_20,           0,      op_22
                                );
        auto assembly = samurai::petsc::make_assembly<bMonolithic>(system_fluid);
        assembly.set_unknowns(velocity, pressure, temperature);
        return assembly;
    }
};

template <class VelocityField, class ScalarField>
auto make_assembly_fluid(VelocityField&, ScalarField&)
{
    return FluidAssembly<VelocityField, ScalarField>();
}

template <class assembly_t>
void set_0_for_all_ghosts_fluid(assembly_t& assembly, Vec v)
{
    assembly.template get<0, 0>().set_0_for_all_ghosts(v);
    assembly.template get<1, 1>().set_0_for_all_ghosts(v);
    assembly.template get<2, 2>().set_0_for_all_ghosts(v);
}

template <class VelocityField, class ScalarField>
struct user_context_fluid
{
    using assembly_t = FluidAssembly<VelocityField, ScalarField>;
    static constexpr std::size_t dim = VelocityField::dim;

    user_context_fluid(VelocityField& _velocity,
                       ScalarField&   _pressure,
                       ScalarField&   _temperature,
                       double& _thermal_diffusivity,
                       double& _Pr,
                       double& _Ra,
                       assembly_t&    _assembly)
                : velocity(_velocity)
                , pressure(_pressure)
                , temperature(_temperature)
                , thermal_diffusivity(_thermal_diffusivity)
                , Pr(_Pr)
                , Ra(_Ra)
                , assembly(_assembly)
    {

    }

    PetscErrorCode update_fields(Vec X)
    {
        PetscFunctionBeginUser;

        auto& mesh = velocity.mesh();
        PetscInt index_shift = 0;

        velocity.fill(0.);
        pressure.fill(0.);
        temperature.fill(0.);

        samurai::petsc::copy(index_shift, X, velocity);
        index_shift += static_cast<PetscInt>(mesh.nb_cells() * dim);

        samurai::petsc::copy(index_shift, X, pressure);
        index_shift += static_cast<PetscInt>(mesh.nb_cells());

        samurai::petsc::copy(index_shift, X, temperature);

        // samurai::update_ghost_mr(velocity, pressure, temperature);

        PetscFunctionReturn(0);
    }
    VelocityField& velocity;
    ScalarField&   pressure;
    ScalarField&   temperature;
    double&        thermal_diffusivity;
    double&        Pr;
    double&        Ra;
    assembly_t&    assembly;
};

template <class VelocityField, class ScalarField, class FluidAssembly>
auto make_user_context_fluid( VelocityField& _velocity,
                              ScalarField&   _pressure,
                              ScalarField&   _temperature,
                              double& _thermal_diffusivity,
                              double& _Pr,
                              double& _Ra,
                              FluidAssembly& _assembly)
{
    return user_context_fluid<VelocityField, ScalarField>(_velocity, _pressure, _temperature, _thermal_diffusivity, _Pr, _Ra, _assembly);
}


static PetscErrorCode IFunction(TS /*ts*/, PetscReal /*t*/, Vec X ,Vec Xdot, Vec F, void* ctx)
{
    PetscFunctionBeginUser;
    
    // std::cout << "Entering IFunction\n";
    
    static constexpr std::size_t dim = 2;
    using Mesh = decltype( samurai::mra::make_empty_mesh( std::declval<decltype( samurai::mesh_config<dim>() )>() ) );
    using ScalarField = samurai::ScalarField<Mesh>;
    using VelocityField = samurai::VectorField<Mesh, double, dim, false>;
    using CTX = user_context_fluid<VelocityField, ScalarField>;

    CTX* user_ctx = reinterpret_cast<CTX*>(ctx);
    
    // std::cout << "Updating fields from X\n";
    user_ctx->update_fields(X);
    samurai::update_ghost_mr(user_ctx->velocity, user_ctx->pressure, user_ctx->temperature);

    auto assembly = user_ctx->assembly.get_assembly<samurai::petsc::BlockAssemblyType::Monolithic>(user_ctx->velocity, user_ctx->pressure, user_ctx->temperature, user_ctx->thermal_diffusivity, user_ctx->Pr, user_ctx->Ra, 0.);

    auto& mesh  = user_ctx->velocity.mesh();

    std::tuple<VelocityField, ScalarField, ScalarField> f_tuple;
    auto& fu = std::get<0>(f_tuple);
    auto& fp = std::get<1>(f_tuple);
    auto& fT = std::get<2>(f_tuple);
    fu = VelocityField("fu", mesh);
    fp = ScalarField("fp", mesh);
    fT = ScalarField("fT", mesh);
    
    // std::cout << "Assembling residual\n";

    // set values of xdot
    // std::cout << "Copying Xdot to local fields\n";
    samurai::petsc::copy(assembly.template get<0,0>().col_shift(), Xdot, fu);
    fp.fill(0.);
    samurai::petsc::copy(assembly.template get<2,2>().col_shift(), Xdot, fT);

    // std::cout << "Computing unknown tuple\n";
    auto unknown_tuple = assembly.unknown();
    assembly.block_operator().apply(f_tuple, unknown_tuple);

    // std::cout << "Copying residual to F\n";
    samurai::petsc::copy(fu, F, assembly.template get<0,0>().col_shift());
    samurai::petsc::copy(fp, F, assembly.template get<1,1>().col_shift());
    samurai::petsc::copy(fT, F, assembly.template get<2,2>().col_shift());

    set_0_for_all_ghosts_fluid(assembly, F);

    PetscFunctionReturn(0);
}

static PetscErrorCode IJacobian(TS /*ts*/, PetscReal /*t*/, Vec /*X*/, Vec /*Xdot*/, PetscReal a, Mat J, Mat P, void* ctx)
{
    PetscFunctionBeginUser;

    // std::cout << "Entering IJacobian\n";
    static constexpr std::size_t dim = 2;
    using Mesh = decltype( samurai::mra::make_empty_mesh( std::declval<decltype( samurai::mesh_config<dim>() )>() ) );
    using ScalarField = samurai::ScalarField<Mesh>;
    using VelocityField = samurai::VectorField<Mesh, double, dim, false>;
    using CTX = user_context_fluid<VelocityField, ScalarField>;

    CTX* user_ctx = reinterpret_cast<CTX*>(ctx);
    
    auto assembly = user_ctx->assembly.get_assembly<samurai::petsc::BlockAssemblyType::Monolithic>(user_ctx->velocity, user_ctx->pressure, user_ctx->temperature, user_ctx->thermal_diffusivity, user_ctx->Pr, user_ctx->Ra, a);

    assembly.setup();
    // assembly.reset();
    // std::cout << "Assembling Jacobian\n";
    MatZeroEntries(P);
    assembly.assemble_matrix(P);
    PetscObjectSetName(reinterpret_cast<PetscObject>(P), "Jacobian");
    if (J != P)
    {
        // assembly.reset();
        // MatZeroEntries(J);
        // assembly.assemble_matrix(J);
        MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY);
    }
    
    // Set the null space (constant pressure) so that iterative solvers can orthogonalize residuals against it
    
    // auto& mesh  = user_ctx->velocity.mesh();
    // SNES snes;
    // KSP ksp;
    // PC pc;
    // TSGetSNES(ts, &snes);
    // SNESGetKSP(snes, &ksp);
    // KSPGetPC(ksp, &pc);

    // auto constant_pressure = samurai::make_scalar_field<double>("constant_pressure", mesh, 1.);
    // auto zero_velocity     = samurai::make_vector_field<double, dim>("zero_velocity", mesh, 0.);
    // auto zero_temperature  = samurai::make_scalar_field<double>("zero_temperature", mesh, 0.);

    // Vec constant_pressure_vector = assembly.create_vector(zero_velocity, constant_pressure, zero_temperature);
    // VecNormalize(constant_pressure_vector, NULL);

    // MatNullSpace nullspace;
    // MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_FALSE, 1, &constant_pressure_vector, &nullspace);
    // MatSetNullSpace(J, nullspace);
    // MatNullSpaceDestroy(&nullspace);
    // VecDestroy(&constant_pressure_vector);

    // // If using MUMPS, set option ICNTL(24)=1 to enable the detection of null pivots (because the matrix is singular)
    // PetscBool is_mumps = PETSC_FALSE;
    // const char* stype;
    // PCFactorGetMatSolverType(pc, &stype);
    // PetscStrcmp(stype, MATSOLVERMUMPS, &is_mumps);
    // if (is_mumps)
    // {
    //     Mat F;
    //     PCFactorGetMatrix(pc, &F);
    //     MatMumpsSetIcntl(F, 24, 1); // (equiv. '-mat_mumps_icntl_24 1')
    // }
    // std::cout << "Jacobian assembly done\n";
    PetscFunctionReturn(0);
}

static PetscErrorCode Register_LMS_ESDIRK5(void)
{
  PetscFunctionBeginUser;

  const PetscReal A[8][8] = {
    {0, 0, 0, 0, 0, 0, 0, 0},

    {1.0/6.0, 1.0/6.0, 0, 0, 0, 0, 0, 0},

    {1.0/24.0, -1.0/24.0, 1.0/6.0, 0, 0, 0, 0, 0},

    {-1.0/6.0, -1.0/3.0, 2.0/3.0, 1.0/6.0, 0, 0, 0, 0},

    {3.0/2.0, 1.0, -10.0/3.0, 5.0/3.0, 1.0/6.0, 0, 0, 0},

    {11473603.0/8957952.0,
     185791.0/124416.0,
     -1757095.0/746496.0,
     199985.0/1990656.0,
     -621575.0/17915904.0,
     1.0/6.0, 0, 0},

    {-17852.0/32289.0,
     -1341.0/1832.0,
     17673.0/11450.0,
     5205.0/84272.0,
     -2471.0/1374000.0,
     456192.0/30943625.0,
     1.0/6.0, 0},

    {-187.0/2820.0,
     0,
     189.0/250.0,
     -135.0/184.0,
     -213.0/5000.0,
     1679616.0/7431875.0,
     229.0/330.0,
     1.0/6.0}
  };

  /* Order 5 weights (stiffly accurate) */
  const PetscReal b[8] = {
    -187.0/2820.0,
    0,
    189.0/250.0,
    -135.0/184.0,
    -213.0/5000.0,
    1679616.0/7431875.0,
    229.0/330.0,
    1.0/6.0
  };

  /* Embedded order 4 weights */
  const PetscReal bembedt[8] = {
    -17852.0/32289.0,
    -1341.0/1832.0,
    17673.0/11450.0,
    5205.0/84272.0,
    -2471.0/1374000.0,
    456192.0/30943625.0,
    1.0/6.0,
    0
  };

  const PetscReal ct[8] = {
    0,
    1.0/3.0,
    1.0/6.0,
    1.0/3.0,
    1.0,
    47.0/72.0,
    1.0/2.0,
    1.0
  };

  PetscCall(TSDIRKRegister("lms_esdirk5",
                          5,                /* order */
                          8,                /* stages */
                          &A[0][0],
                          b,
                          ct,
                          bembedt,
                          0,
                          PETSC_NULLPTR));

  PetscFunctionReturn(PETSC_SUCCESS);
}


template <class VelocityField, class ScalarField>
inline void configure_and_solve_fluid_ts(VelocityField &velocity,
                                     ScalarField &pressure,
                                     ScalarField &temperature,
                                     double& thermal_diffusivity,
                                     double& Pr,
                                     double& Ra,
                                     double& dt,
                                     double t)
{
    samurai::update_ghost_mr(velocity, pressure, temperature);
    TS ts;
    SNES snes;
    KSP ksp;
    PC pc;
    Mat J;
    Vec X;

    // PetscBool bJacEvalLag  = PETSC_TRUE;
    // PetscInt  jac_eval_lag = 1;
    // PetscBool bPCEvalLag   = PETSC_TRUE;
    // PetscInt  pc_eval_lag  = 1;
    std::cout << "Configuring TS solver at t = " << t << "\n";

    std::cout << "Creating assembly object\n";
    auto assembly_obj = make_assembly_fluid( velocity, pressure);

    auto assembly = assembly_obj.template get_assembly<samurai::petsc::BlockAssemblyType::Monolithic>( velocity, pressure, temperature, thermal_diffusivity, Pr, Ra, 0.);

    std::cout << "Creating user context\n";
    auto user_ctx = make_user_context_fluid( velocity, pressure, temperature, thermal_diffusivity, Pr, Ra, assembly_obj);

    std::cout << "Creating Jacobian matrix\n";
    assembly.create_matrix(J);
    MatSetOptionsPrefix(J, "J_");
    MatSetFromOptions(J);

    std::cout << "Creating solution vector\n";
    X = assembly.create_vector(velocity, pressure, temperature);

    // create TS solver
    std::cout << "Creating TS solver\n";
    TSCreate(PETSC_COMM_SELF, &ts);
    // TSSetProblemType(ts, TS_NONLINEAR);
    TSSetEquationType(ts, TS_EQ_DAE_SEMI_EXPLICIT_INDEX2);

    std::cout << "Configuring TS solver type\n";
    TSSetType(ts, TSARKIMEX);
    TSARKIMEXSetType(ts, TSARKIMEX2E);
    // TSSetType(ts, TSDIRK);
    // TSDIRKSetType(ts, "lms_esdirk5");

    std::cout << "Setting TS time and time step\n";
    TSSetTime(ts, t);
    TSSetTimeStep(ts, dt);
    TSSetMaxSteps(ts, 10000);

    std::cout << "Setting TS max time\n";
    TSSetMaxTime(ts, t + dt);
    // TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER);
    TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP);

    // setup SNES solver
    std::cout << "Setting up IFunction \n";
    TSSetIFunction(ts, PETSC_NULLPTR, IFunction, &user_ctx);
    std::cout << "Setting up IJacobian \n";
    TSSetIJacobian(ts, J, J, IJacobian, &user_ctx);

    std::cout << "Setting TS solution\n";
    TSSetSolution(ts, X);

    // Configure TS solver
    std::cout << "Configuring TS solver [snes, ksp, pc]\n";
    TSGetSNES(ts, &snes);
    SNESGetKSP(snes, &ksp);
    KSPGetPC(ksp, &pc);
    
    SNESLineSearch linesearch;
    SNESGetLineSearch(snes, &linesearch);
    SNESLineSearchSetType(linesearch, SNESLINESEARCHCP); // (equiv. '-snes_linesearch_type cp')
    
    // We use MUMPS because it can handle null spaces (unlike the default LU solver)
    KSPSetType(ksp, KSPPREONLY);                  // (equiv. '-ksp_type preonly')
    PCSetType(pc, PCLU);                          // (equiv. '-pc_type lu')
    PCFactorSetMatSolverType(pc, MATSOLVERMUMPS); // (equiv. '-pc_factor_mat_solver_type mumps')
    
    // We set the same tolerance as that of the multiresolution
    PetscReal atol = temperature.mesh().min_level() == temperature.mesh().max_level() ? 1e-6 : 1e-3;
    SNESSetTolerances(snes, atol, PETSC_DETERMINE, PETSC_DETERMINE, PETSC_DETERMINE, PETSC_DETERMINE); // (equiv. '-snes_atol [atol]')

    TSSetFromOptions(ts);
    SNESSetFromOptions(snes);
    KSPSetFromOptions(ksp);
    PCSetFromOptions(pc);

    // solve the time step
    std::cout << "Solving TS step\n";
    TSSolve(ts, PETSC_NULLPTR);

    TSConvergedReason reason_code;
    TSGetConvergedReason(ts, &reason_code);
    if (reason_code < 0)
    {
        assert(false && "Divergence of the TS solver");
        exit(EXIT_FAILURE);
    }

    // std::cout << "Solution:"
    //           << "\nVelocity field:\n" << velocity
    //           << "\nPressure field:\n" << pressure
    //           << "\nTemperature field:\n" << temperature
    //           << "\n-------\n-------" << std::endl;


    VecDestroy(&X);
    MatDestroy(&J);
    TSDestroy(&ts);
}
