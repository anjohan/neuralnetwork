module mod_activation_functions
    use iso_fortran_env, only: dp => real64
    implicit none

    type, abstract :: activation_function
        contains
            procedure(diff_procedure), deferred, nopass :: diff
            procedure(eval_procedure), deferred, nopass :: eval
    end type

    type, extends(activation_function) :: sigmoid
        contains
            procedure, nopass :: diff => diff_sigmoid
            procedure, nopass :: eval => eval_sigmoid
    end type

    type, extends(activation_function) :: relu
        contains
            procedure, nopass :: diff => diff_relu
            procedure, nopass :: eval => eval_relu
    end type

    abstract interface
        subroutine eval_procedure(input, output)
            import dp
            real(dp), intent(in) :: input(:)
            real(dp), intent(out) :: output(:)
        end subroutine
        subroutine diff_procedure(eval_output, diff_output)
            import dp
            real(dp), intent(in) :: eval_output(:)
            real(dp), intent(out) :: diff_output(:)
        end subroutine
    end interface

    contains
        subroutine eval_sigmoid(input, output)
            real(dp), intent(in) :: input(:)
            real(dp), intent(out) :: output(:)

            output(:) = 1/(1+exp(-input(:)))
        end subroutine

        subroutine diff_sigmoid(eval_output, diff_output)
            real(dp), intent(in) :: eval_output(:)
            real(dp), intent(out) :: diff_output(:)

            diff_output(:) = eval_output(:) * (1 - eval_output(:))
        end subroutine

        subroutine eval_relu(input, output)
            real(dp), intent(in) :: input(:)
            real(dp), intent(out) :: output(:)

            where (input >= 0)
                output = input
            else where
                output = 0
            end where
        end subroutine

        subroutine diff_relu(eval_output, diff_output)
            real(dp), intent(in) :: eval_output(:)
            real(dp), intent(out) :: diff_output(:)

            where (eval_output >= 0)
                diff_output = 1
            else where
                diff_output = 0
            end where
        end subroutine
end module
