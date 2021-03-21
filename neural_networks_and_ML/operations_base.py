class Operation:
    def __init__(self):
        self.input = None
        self.grad_input = None
        self.output = None
        self.grad_output = None
        self.training_run = False

    def forward(self, input_, training_run=False):
        self.input = input_
        self.training_run = training_run
        self.output = self.eval_output()
        return self.output

    def backward(self, grad_output):
        self.grad_output = grad_output
        self.grad_input = self.eval_input_gradient()
        return self.grad_input

    def eval_output(self):
        raise NotImplementedError

    def eval_input_gradient(self):
        raise NotImplementedError


class ParameterOperation(Operation):
    def __init__(self, parameter_matrix):
        super().__init__()
        self.parameter_matrix = parameter_matrix
        self.grad_parameters = None

    def backward(self, grad_output):
        self.grad_output = grad_output
        self.grad_input = self.eval_input_gradient()
        self.grad_parameters = self.eval_parameter_gradient()
        return self.grad_input

    def eval_parameter_gradient(self):
        raise NotImplementedError
