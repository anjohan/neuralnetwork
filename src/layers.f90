module mod_layers
    use mod_activation_functions
    use iso_fortran_env, only: dp => real64
    implicit none

    type :: layer
        real(dp), allocatable :: W(:,:), b(:), delta(:), output(:), &
                                 grad_W(:,:), grad_b(:), z(:), f_diff(:), input(:)
        class(activation_function), allocatable :: f

        contains
            procedure :: init
            procedure :: predict, update_weights, zero_grads, update_grads
            generic :: back_prop => hidden_back_prop, output_back_prop
            procedure :: hidden_back_prop, output_back_prop
    end type

    contains
        subroutine init(self, num_inputs, num_outputs, f)
            class(layer), intent(out) :: self
            integer, intent(in) :: num_inputs, num_outputs
            class(activation_function), intent(in), optional :: f

            ! no f => regression layer
            if (present(f)) self%f = f

            allocate(self%W(num_outputs, num_inputs), self%z(num_outputs))
            allocate(self%input(num_inputs))

            allocate(self%grad_W, mold=self%W)
            allocate(self%b, self%grad_b, self%output, &
                     self%delta, self%f_diff, &
                     mold=self%z)

            call random_number(self%W)
            self%W(:,:) = (self%W(:,:)-0.5d0) / sqrt(1.0d0*num_inputs)
            self%b(:) = 0

            call self%zero_grads()
        end subroutine

        subroutine predict(self, input)
            class(layer), intent(inout) :: self
            real(dp), intent(in) :: input(:)

            self%z(:) = matmul(self%W, input)
            self%z(:) = self%z(:) + self%b(:)
            self%input(:) = input(:)

            if (allocated(self%f)) then
                call self%f%eval_diff(self%z, self%output, self%f_diff)
            else
                self%output(:) = self%z(:)
                self%f_diff(:) = 1
            end if
        end subroutine

        subroutine hidden_back_prop(self, next_layer)
            !! back propagation in non-output layer, calculate delta

            class(layer), intent(inout) :: self
            class(layer), intent(in) :: next_layer

            integer :: j

            do j = 1, size(self%delta)
                self%delta(j) = dot_product(next_layer%delta, next_layer%W(:,j)) &
                                * self%f_diff(j)
            end do

            call self%update_grads()

        end subroutine

        subroutine output_back_prop(self, y)
            !! back propagation in output layer, calculate delta

            class(layer), intent(inout) :: self
            real(dp), intent(in) :: y(:)

            self%delta(:) = self%output(:) - y(:)
            !write(*,*) self%output(:), y, self%delta
            call self%update_grads()

        end subroutine

        subroutine update_grads(self)
            !! update gradients (after calculating delta)

            class(layer), intent(inout) :: self
            integer :: k

            self%grad_b(:) = self%grad_b(:) + self%delta

            do k = 1, size(self%W, 2)
                self%grad_W(:,k) = self%grad_W(:,k) + self%delta(:) * self%input(k)
            end do
        end subroutine

        subroutine update_weights(self, learning_rate, lambda)
            !! update weights using gradient averaged over batch

            class(layer), intent(inout) :: self
            real(dp), intent(in) :: learning_rate, lambda

            if (num_images() > 1) then
                call co_sum(self%grad_W)
                call co_sum(self%grad_b)
            end if

            if (lambda /= 0) then
                self%grad_W(:,:) = self%grad_W(:,:) + lambda*self%W(:,:)
                self%grad_b(:) = self%grad_b(:) + lambda*self%b(:)
            end if

            self%W(:,:) = self%W(:,:) - learning_rate*self%grad_W(:,:)
            self%b(:) = self%b(:) - learning_rate*self%grad_b(:)

            call self%zero_grads()
        end subroutine

        subroutine zero_grads(self)
            !! set gradients to zero before next batch

            class(layer), intent(inout) :: self

            self%grad_W(:,:) = 0
            self%grad_b(:) = 0
        end subroutine
end module
