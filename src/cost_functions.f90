module mod_cost_functions
    use iso_fortran_env, only: dp => real64
    implicit none

    abstract interface
        function cost_function(prediction, y) result(cost)
            import dp
            real(dp) :: cost
            real(dp), intent(in) :: prediction(:), y(:)
        end function
    end interface

    contains

        function mean_squared_error(prediction, y) result(cost)
            real(dp) :: cost
            real(dp), intent(in) :: prediction(:), y(:)

            cost = 0.5*sum((prediction-y)**2)/size(y)
        end function

        function crossentropy(prediction, y) result(cost)
            real(dp) :: cost
            real(dp), intent(in) :: prediction(:), y(:)

            if (size(y) /= size(prediction)) error stop

            if (size(y) == 1) then
                cost =  -y(1)*log(prediction(1)) - (1-y(1))*log(1-prediction(1))
            else
                cost = -sum(y(:) * log(prediction(:)))
            end if
        end function

end module
