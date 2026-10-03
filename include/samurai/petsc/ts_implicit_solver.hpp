#pragma once
#include "fv/cell_based_scheme_assembly.hpp"
#include "fv/flux_based_scheme_assembly.hpp"
#include "fv/operator_sum_assembly.hpp"
#include "utils.hpp"
#include <petsc.h>

namespace samurai
{
    namespace petsc
    {
        template <class Assembly>
        class TSImplicitSolverBase
        {
            using scheme_t = typename Assembly::scheme_t;

            static constexpr bool is_block_solver = IsBlockOperator<scheme_t>;

            using output_field_t = typename scheme_t::output_field_t;

            // Helper to conditionally access output_field_no_ref_t only when it exists
            template <class Scheme, bool is_block>
            struct worker_output_field_type_helper
            {
                using type = output_field_t;
            };

            template <class Scheme>
            struct worker_output_field_type_helper<Scheme, true>
            {
                using type = typename Scheme::output_field_no_ref_t;
            };

            template <class Scheme>
            struct worker_xdot_field_type_helper<Scheme, true>
            {
                using type = typename Scheme::output_field_no_ref_t;
            };

            // If not a block solver, worker_output_field_t = output_field_t
            using worker_output_field_t = typename worker_output_field_type_helper<scheme_t, is_block_solver>::type;

            // If not a block solver, worker_xdot_field_t = output_field_t
            using worker_xdot_field_t = typename worker_xdot_field_type_helper<scheme_t, is_block_solver>::type;

          protected:

            Assembly m_assembly;
            TS m_ts                           = nullptr;
            Mat m_J                           = nullptr;
            bool m_is_set_up                  = false;
            bool m_reuse_allocated_matrix     = false;
            bool m_stop_program_on_divergence = true;
            double m_time_step_inverse        = 0.;

            worker_output_field_t m_worker_output_field;
            worker_xdot_field_t m_worker_xdot_field;

          public:

            // User callback to configure the solver
            std::function<void(TS&, SNES&, KSP&, PC&)> configure                                = nullptr;
            std::function<void(TS&, SNES&, KSP&, PC&, Mat&)> after_matrix_assembly              = nullptr;
            std::function<void(PetscInt, PetscReal, const worker_output_field_t&)> monitor = nullptr;

            explicit TSImplicitSolverBase(const scheme_t& scheme)
                : m_assembly(scheme)
            {
                TSCreate(PETSC_COMM_WORLD, &m_ts);
            }

            virtual ~TSImplicitSolverBase()
            {
                _destroy_petsc_objects();
            }

          private:

            void _destroy_petsc_objects()
            {
                if (m_J)
                {
                    m_assembly.destroy_local_to_global_mappings(m_J);
                    MatDestroy(&m_J);
                    m_J                      = nullptr;
                    m_reuse_allocated_matrix = false;
                }
                if (m_ts)
                {
                    TSDestroy(&m_ts);
                    m_ts = nullptr;
                }
            }

          public:

            virtual void destroy_petsc_objects()
            {
                _destroy_petsc_objects();
            }

            TSImplicitSolverBase& operator=(const TSImplicitSolverBase& other)
            {
                if (this != &other)
                {
                    this->destroy_petsc_objects();
                    this->m_assembly  = other.m_assembly;
                    this->m_ts        = other.m_ts;
                    this->m_J         = other.m_J;
                    this->m_is_set_up = other.m_is_set_up;
                }
                return *this;
            }

            TSImplicitSolverBase& operator=(TSImplicitSolverBase&& other)
            {
                if (this != &other)
                {
                    this->destroy_petsc_objects();
                    this->m_assembly  = other.m_assembly;
                    this->m_ts        = other.m_ts;
                    this->m_J         = other.m_J;
                    this->m_is_set_up = other.m_is_set_up;
                    other.m_ts        = nullptr; // Prevent TS destruction when 'other' object is destroyed
                    other.m_J         = nullptr;
                    other.m_is_set_up = false;
                }
                return *this;
            }

            TS& Ts()
            {
                return m_ts;
            }

            bool is_set_up()
            {
                return m_is_set_up;
            }

            auto& assembly()
            {
                return m_assembly;
            }

            auto& scheme()
            {
                return assembly().scheme();
            }

            void stop_program_on_divergence(bool value)
            {
                m_stop_program_on_divergence = value;
            }

          protected:

            virtual void default_solver_configuration()
            {
            }

          public:

            virtual void setup()
            {
                if (is_set_up())
                {
                    return;
                }

                if (assembly().undefined_unknown())
                {
                    std::cerr << "Undefined unknown(s) for this ts implicit system. Please set the unknowns using the instruction '[solver].set_unknown(u);' or '[solver].set_unknowns(u1, u2...);'."
                              << std::endl;
                    assert(false && "Undefined unknown(s)");
                    exit(EXIT_FAILURE);
                }

                // Implicit TS function
                TSSetIFunction(m_ts, nullptr, PETSC_ts_implicit_function, this);

                // Jacobian matrix
                if (!m_reuse_allocated_matrix)
                {
                    assembly().create_matrix(m_J);
                }

                // Jacobian function
                TSSetIJacobian(m_ts, m_J, m_J, PETSC_ts_implicit_jacobian_function, this);

                // Sends the successive residuals to the user if he monitors the convergence
                if (monitor)
                {
                    TSMonitorSet(m_ts, PETSC_monitor, this, nullptr);
                }

                SNES snes;
                KSP ksp;
                PC pc;
                TSGetSNES(m_ts, &snes);
                SNESGetKSP(snes, &ksp);
                KSPGetPC(ksp, &pc);
                if (configure)
                {
                    configure(m_ts, snes, ksp, pc);
                }
                TSSetFromOptions(m_ts);

                m_is_set_up = true;
            }

          private:

            static PetscErrorCode PETSC_ts_implicit_function(TS /*ts*/, PetscReal t, Vec x, Vec xdot, Vec f, void* ctx)
            {
                times::timers.stop("ts implicit system solve");

                auto self      = reinterpret_cast<TSImplicitSolverBase*>(ctx); // this
                auto& assembly = self->assembly();

                if constexpr (!is_block_solver)
                {
                    // Ideally, we would like to wrap a field structure around the data of the Petsc vectors x and f,
                    // but we don't have such a Field constructor.
                    // So, instead, we use worker fields and copy the data.
                    auto& x_field = assembly.unknown();          // for x, we reuse the unknown field
                    auto& f_field = self->m_worker_output_field; // for f, we use an actual worker field
                    auto& xdot_field = self->m_worker_xdot_field; // for xdot, we use an actual worker field

                    assembly.copy_unknown(x, x_field);
                    assembly.copy_unknown(xdot, xdot_field);

                    // Apply explicit scheme: f = scheme(x)
                    f_field.fill(0); // initialize to zero because we accumulate the results
                    self->scheme().apply(f_field, x_field);

                    f_field += xdot_field; // Add the contribution from the time derivative

                    // Copy the result into the Petsc vector f
                    assembly.copy_rhs(f_field, f);
                    // Set to zero the right-hand side of the ghost equations and apply BCs attached to the unknown field
                    self->prepare_rhs(x, f);
                }
                else
                {
                    assembly.update_unknowns(x);                  // for x, we reuse the unknown fields
                    auto& f_fields = self->m_worker_output_field; // for f, we use an actual worker field, which is a tuple of fields
                    auto& xdot_fields = self->m_worker_xdot_field; // for xdot, we use an actual worker field, which is a tuple of fields

                    // Apply explicit scheme: f = scheme(x)
                    for_each(f_fields,
                             [](auto& f_field)
                             {
                                 f_field.fill(0); // initialize to zero because we accumulate the results
                             });

                    auto unknown_tuple = assembly.unknown(); // tuple containing references to the unknown fields
                    
                    xdot_fields = assembly.copy_unknowns(xdot, xdot_fields);

                    assembly.block_operator().apply(f_fields, unknown_tuple);

                    for_each_assembly_op(
                        [&](auto& op, auto row, auto col)
                        {
                            if constexpr (row == col)
                            {
                                auto& f_field = std::get<row>(f_fields);
                                f_field += std::get<row>(xdot_fields);
                            }
                        });
                    // Copy the result into the Petsc vector f
                    assembly.copy_rhs(f_fields, f);
                    // Set to zero the right-hand side of the ghost equations and apply BCs attached to the unknown field
                    self->prepare_rhs(x, f);
                }

#ifdef SAMURAI_CHECK_NAN
                assert(check_nan_or_inf(f));
#endif
                times::timers.start("ts implicit system solve");
                return PETSC_SUCCESS;
            }

            static PetscErrorCode PETSC_ts_implicit_jacobian_function(SNES snes, Vec x, Mat jac, Mat B, void* ctx)
            {
                times::timers.stop("ts implicit system solve");

                // Here, jac = B = this.m_J

                auto self      = reinterpret_cast<TSImplicitSolverBase*>(ctx); // this
                auto& assembly = self->assembly();

                // Ideally, we would like to wrap a field structure around the data of the Petsc vector x,
                // but we don't have such a Field constructor.
                // So, instead, we reuse the unknown field and copy the data.
                if constexpr (!is_block_solver)
                {
                    assembly.copy_unknown(x, assembly.unknown());
                }
                else
                {
                    assembly.update_unknowns(x);
                }

                // Assembly of the Jacobian matrix.
                // In this case, jac = B, but Petsc recommends we assemble B for more general cases.
                MatZeroEntries(B);
                assembly.assemble_matrix(B);
                PetscObjectSetName(reinterpret_cast<PetscObject>(B), "Jacobian");
                if (jac != B)
                {
                    MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY);
                    MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY);
                }

                if (self->after_matrix_assembly)
                {
                    KSP ksp;
                    PC pc;
                    SNESGetKSP(snes, &ksp);
                    KSPGetPC(ksp, &pc);
                    self->after_matrix_assembly(snes, ksp, pc, B);
                }

                // MatView(B, PETSC_VIEWER_STDOUT_(PETSC_COMM_WORLD));
                // std::cout << std::endl;

                times::timers.start("ts implicit system solve");
                return PETSC_SUCCESS;
            }

            static PetscErrorCode PETSC_monitor(SNES snes, PetscInt it, PetscReal rnorm, void* ctx)
            {
                Vec r;
                SNESGetFunction(snes, &r, NULL, NULL);

                auto self      = reinterpret_cast<TSImplicitSolverBase*>(ctx); // this
                auto& assembly = self->assembly();

                auto& r_field = self->m_worker_output_field;
                assembly.copy_rhs(r, r_field);
                assert(self->monitor);
                self->monitor(it, rnorm, r_field);

                return PETSC_SUCCESS;
            }

          protected:

            void prepare_rhs(Vec& /* x */, Vec& b)
            {
                // assembly().copy_values_for_all_ghosts(x, b);
                assembly().set_0_for_all_ghosts(b);
                // Update the right-hand side with the boundary conditions stored in the solution field
                assembly().enforce_bc(b);
                // Set to zero the right-hand side of the ghost equations
                // assembly().enforce_projection_prediction(b);
                // Set to zero the right-hand side of the useless ghosts' equations
                // assembly().set_0_for_useless_ghosts(b);

                // VecView(b, PETSC_VIEWER_STDOUT_(PETSC_COMM_WORLD));
                // std::cout << std::endl;
                // assert(check_nan_or_inf(b));
            }

            SNESConvergedReason solve_system(Vec& x, const Vec& b)
            {
#ifdef SAMURAI_CHECK_NAN
                assert(check_nan_or_inf(x));
                assert(check_nan_or_inf(b));
#endif
                // Solve the system
                times::timers.start("ts implicit system solve");
                SNESSolve(m_snes, b, x);
                times::timers.stop("ts implicit system solve");

                SNESConvergedReason reason_code;
                SNESGetConvergedReason(m_snes, &reason_code);
                if (reason_code < 0 && m_stop_program_on_divergence)
                {
                    using namespace std::string_literals;
                    const char* reason_text;
                    SNESGetConvergedReasonString(m_snes, &reason_text);
                    std::cerr << "Divergence of the non-linear solver ("s + reason_text + ")" << std::endl;
                    // VecView(b, PETSC_VIEWER_STDOUT_(PETSC_COMM_WORLD));
                    // std::cout << std::endl;
                    // assert(check_nan_or_inf(b));
                    assert(false && "Divergence of the solver");
                    exit(EXIT_FAILURE);
                }
                // VecView(x, PETSC_VIEWER_STDOUT_(PETSC_COMM_WORLD)); std::cout << std::endl;
                return reason_code;
            }

          public:

            int iterations()
            {
                PetscInt n_iterations;
                SNESGetIterationNumber(m_snes, &n_iterations);
                return n_iterations;
            }

            virtual void reset()
            {
                destroy_petsc_objects();
                SNESCreate(PETSC_COMM_WORLD, &m_snes);
                m_assembly.is_set_up(false);
                m_is_set_up = false;
                default_solver_configuration();
            }

            void set_scheme(const scheme_t& s)
            {
                m_assembly.set_scheme(s);
                if (m_snes)
                {
                    SNESDestroy(&m_snes);
                    SNESCreate(PETSC_COMM_WORLD, &m_snes);
                }
                default_solver_configuration();
                m_is_set_up              = false;
                m_reuse_allocated_matrix = m_J != nullptr;
            }
        };

        template <class Scheme>
        class NonLinearSolver : public NonLinearSolverBase<Assembly<Scheme>>
        {
            using base_class = NonLinearSolverBase<Assembly<Scheme>>;

          public:

            using scheme_t       = Scheme;
            using input_field_t  = typename scheme_t::input_field_t;
            using output_field_t = typename scheme_t::output_field_t;
            using Mesh           = typename input_field_t::mesh_t;

            using base_class::assembly;
            using base_class::m_is_set_up;
            using base_class::m_J;
            using base_class::m_worker_output_field;

            explicit NonLinearSolver(const scheme_t& scheme)
                : base_class(scheme)
            {
            }

            void set_unknown(input_field_t& unknown)
            {
                assembly().set_unknown(unknown);
            }

            SNESConvergedReason solve(output_field_t& rhs)
            {
                m_worker_output_field = output_field_t("worker_output", rhs.mesh());

                if (!m_is_set_up)
                {
                    this->setup();
                }

                Vec b = assembly().create_rhs_vector(rhs);
                Vec x = assembly().create_solution_vector(assembly().unknown());

                this->prepare_rhs(x, b);

                SNESConvergedReason reason_code = this->solve_system(x, b);

#ifdef SAMURAI_WITH_MPI
                assembly().copy_unknown(x, assembly().unknown());
#endif
                VecDestroy(&b);
                VecDestroy(&x);
                return reason_code;
            }

            void solve(input_field_t& unknown, output_field_t& rhs)
            {
                set_unknown(unknown);
                solve(rhs);
            }
        };

    } // end namespace petsc
} // end namespace samurai
