module mod_neural_network
    use mod_activation_functions
    use mod_layers
    use mod_cost_functions
    use iso_fortran_env, only: dp => real64
    implicit none

    interface neural_network
        procedure :: init_neural_network
    end interface

    type :: neural_network
        class(layer), allocatable :: layers(:)
        integer :: num_layers
        real(dp) :: lambda

        contains
            procedure :: predict => nn_predict, feed_forward, cost_func
            procedure :: back_prop => nn_back_prop
            procedure :: train, update_weights => nn_update_weights
            procedure :: reset_weights => nn_reset_weights
    end type

    contains
        function init_neural_network(num_inputs, nums_neurons, &
                                     hidden_activation, output_activation, lambda) &
                 result(self)
            !! constructor of a neural network

            type(neural_network) :: self
            integer, value :: num_inputs
            integer, intent(in) :: nums_neurons(:)
                !! number of neurons in (= number of outputs from) each layer
            class(activation_function), intent(in) :: hidden_activation
            class(activation_function), intent(in), optional :: output_activation
            real(dp), intent(in), optional :: lambda

            integer :: num_layers, l
            num_layers = size(nums_neurons)
            self%num_layers = num_layers

            if (present(lambda)) then
                self%lambda = lambda
            else
                self%lambda = 0
            end if

            allocate(self%layers(num_layers))
            associate(layers => self%layers)
                do l = 1, num_layers-1
                    call layers(l)%init(num_inputs, nums_neurons(l), hidden_activation)
                    num_inputs = nums_neurons(l)
                end do

                if (present(output_activation)) then
                    call layers(num_layers)%init(num_inputs, &
                                                 nums_neurons(num_layers), &
                                                 output_activation)
                else
                    call layers(num_layers)%init(num_inputs, &
                                                 nums_neurons(num_layers))
                end if
            end associate
        end function

        subroutine train(self, X, Y, learning_rate, num_epochs, batch_size)
            class(neural_network), intent(inout) :: self
            real(dp), intent(in) :: X(:,:), Y(:,:), learning_rate
            integer, intent(in) :: num_epochs, batch_size

            integer :: num_inputs, num_batches, i, j, k, idx
            real(dp), allocatable :: rnd(:)

            num_inputs = size(X,2)
            if (size(Y,2) /= num_inputs) error stop
            num_batches = num_inputs/batch_size
            allocate(rnd(batch_size/num_images()))

            do i = 1, num_epochs
                do j = 1, num_batches
                    call random_number(rnd)
                    do k = 1, batch_size/num_images()
                        idx = floor(num_inputs*rnd(k)) + 1
                        call self%feed_forward(X(:,idx))
                        call self%back_prop(Y(:,idx))
                    end do
                    call self%update_weights(learning_rate/batch_size)
                end do
            end do
        end subroutine

        function cost_func(self, X, Y, cost) result(tmp_cost)
            class(neural_network), intent(inout) :: self
            real(dp), intent(in) :: X(:,:), Y(:,:)
            procedure(cost_function) :: cost

            integer :: j
            real(dp) :: tmp_cost

            tmp_cost = 0

            do j = 1, size(X,2)
                call self%feed_forward(X(:,j))
                tmp_cost = tmp_cost + cost(self%layers(self%num_layers)%output, Y(:,j))
            end do
            !write(*,*) tmp_cost
        end function

        subroutine nn_update_weights(self, learning_rate)
            class(neural_network), intent(inout) :: self
            real(dp), intent(in) :: learning_rate

            integer :: l

            do l = 1, self%num_layers
                call self%layers(l)%update_weights(learning_rate, self%lambda)
            end do
        end subroutine

        subroutine nn_predict(self, input, output)
            class(neural_network), intent(inout) :: self
            real(dp), intent(in) :: input(:)
            real(dp), intent(out) :: output(:)

            call self%feed_forward(input)
            output(:) = self%layers(self%num_layers)%output(:)
        end subroutine

        subroutine feed_forward(self, input)
            class(neural_network), intent(inout) :: self
            real(dp), intent(in) :: input(:)

            integer :: l

            associate(layers => self%layers)
                call layers(1)%predict(input)
                do l = 2, self%num_layers
                    call layers(l)%predict(layers(l-1)%output)
                end do
            end associate
        end subroutine

        subroutine nn_back_prop(self, output)
            class(neural_network), intent(inout) :: self
            real(dp), intent(in) :: output(:)

            integer :: l
            associate(layers => self%layers, &
                      num_layers => self%num_layers)
                call layers(num_layers)%back_prop(output)
                do l = num_layers-1, 1, -1
                    call layers(l)%back_prop(layers(l+1))
                end do
            end associate
        end subroutine

        subroutine nn_reset_weights(self)
            class(neural_network), intent(inout) :: self

            integer :: l
            do l = 1, self%num_layers
                call self%layers(l)%reset_weights()
            end do
        end subroutine
end module
