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

        function binary_crossentropy(prediction, y) result(cost)
            real(dp) :: cost
            real(dp), intent(in) :: prediction(:), y(:)

            if(size(prediction) /= 1 .or. size(y) /= 1) error stop

            cost =  -y(1)*log(prediction(1)) - (1-y(1))*log(1-prediction(1))
        end function

end module
