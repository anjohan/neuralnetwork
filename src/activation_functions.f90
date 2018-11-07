module mod_activation_functions
    use iso_fortran_env, only: dp => real64
    implicit none

    type, abstract :: activation_function
        contains
            procedure(eval_diff_procedure), deferred, nopass :: eval_diff
    end type

    type, extends(activation_function) :: sigmoid
        contains
            procedure, nopass :: eval_diff => eval_diff_sigmoid
    end type

    type, extends(activation_function) :: relu
        contains
            procedure, nopass :: eval_diff => eval_diff_relu
    end type

    abstract interface
        subroutine eval_diff_procedure(input, output, diff_output)
            import dp
            real(dp), intent(in) :: input(:)
            real(dp), intent(out) :: output(:), diff_output(:)
        end subroutine
    end interface

    contains
        subroutine eval_diff_sigmoid(input, output, diff_output)
            real(dp), intent(in) :: input(:)
            real(dp), intent(out) :: output(:)
            real(dp), intent(out) :: diff_output(:)

            output(:) = 1/(1+exp(-input(:)))
            diff_output(:) = output(:) * (1 - output(:))
        end subroutine

        subroutine eval_diff_relu(input, output, diff_output)
            real(dp), intent(in) :: input(:)
            real(dp), intent(out) :: output(:)
            real(dp), intent(out) :: diff_output(:)

            where (input >= 0)
                output = input
                diff_output = 1
            elsewhere
                output = 0
                diff_output = 0
            end where
        end subroutine
end module
