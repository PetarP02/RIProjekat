import copy as cp
import math
from typing import Union

# n - number of verices
        # init : O(n)
        # setOperation : O(log n) 
        # setOperand : O(log n) 
        # __replaceNode : O(log n)
        # valueCalc : O(log n) : eval(str(self.__operand[0])) is O(N) N-number of caracters that need to be evaluated
        #                        complexity could be O(N log n) but is N is small most of times.
        # subTree : O(log n)
        # getOperation : O(1)
        # getLeaves : O(1)
        # __str__ : O(n)

class Node:
    """
    Represents a node in a binary expression tree for mathematical expressions.

    Each Node can be:
        - A leaf node containing a single operand (number or variable), or
        - An operator node combining one or two child nodes.

    Features:
        - Supports unary operations: sin, cos, log, exp
        - Supports binary operations: +, -, *, /, ** 
        - Handles both numeric constants and symbolic variables
        - Can evaluate expressions directly or with variable assignments

    Attributes:
        value (float | None): The evaluated value of the expression at this node,
            or None if the expression contains variables.
        size (int): The number of nodes in the subtree rooted at this node.

    Methods (main):
        setOperation(newOperation): Sets a new operation and recalculates.
        setOperand(newOperand, operandPos=0): Replaces an operand and recalculates.
        valueCalc(): Recalculates the value of the expression tree.
        valueCalcVar(varVal): Evaluates the tree with variable assignments.
        subTree(pos, insertTree=None): Retrieves or replaces a subtree.
        getLeaves(): Returns leaf values (constants or variable names).
        getOperation(): Returns the operator of this node.
        nodeInfo(): Prints a readable summary of the node.

    Notes:
        - Division by zero raises an error.
    """

    __variableIndicator = "__nodeVar__"
    __unaryOperations = ['sin', 'cos', 'log', 'exp']
    __binaryOperations = ['+', '-', '*', '/', '**']

    __mapUnary = {'sin' : lambda x: math.sin(x),
                  'cos' : lambda x: math.cos(x), 
                  'log' : lambda x: math.log(x), 
                  'exp' : lambda x: math.exp(x)}
    
    def __init__(self, first: Union['Node', int, str], operation: str = None, second: Union['Node', int, str] = None) -> 'Node':
        """
        Initializes a Node in the binary expression tree.
    
        Args:
            first (Union[Node, int, str]): The first operand or left subtree.
                - If 'operation' is None, this becomes a leaf node containing 'first'.
                - If 'operation' is provided, 'first' becomes the left child node.
            operation (str, optional): The operator for the node. Supported operators:
                - Binary: '+', '-', '*', '/', '**'
                - Unary: 'sin', 'cos', 'log', 'exp'
                Defaults to None.
            second (Union[Node, int, str], optional): The second operand or right subtree.
                Required if a binary operation is provided.
    
        Raises:
            AttributeError: If a binary operation is provided without 'second'.
            AttributeError: If 'operation' is not supported.
        """
        
        if operation in Node.__binaryOperations and second is None:
            raise AttributeError(f"Operations {Node.__binaryOperations} must have second operand")
        if operation is not None and operation not in (Node.__binaryOperations + Node.__unaryOperations):
            raise AttributeError(
                f"Operation {operation} is not supported, supported operations: \n {Node.__binaryOperations + Node.__unaryOperations}")
        
        if operation in Node.__binaryOperations:
            self.op = 2
        elif operation in Node.__unaryOperations:
            self.op = 1
        elif operation is None and second is None:
            self.op = 0

        self.__parent = None
        self.__operator = None
        self.__operand = []
        self.__leaves = set()
        self.size = 1
        self.value = None
        
        if self.op == 0:
            self.__operand.append(first if Node.__isNumber(first) else Node.__variableIndicator + str(first) + Node.__variableIndicator) 
            self.valueCalc()
        elif self.op == 2:
            self.__operator = operation
            left = first if isinstance(first, Node) else Node(first)
            right = second if isinstance(second, Node) else Node(second)
            left.__parent = self
            right.__parent = self
            self.__operand = [left, right]
            self.size = 1 + left.size + right.size
            self.valueCalc()
        elif self.op == 1:
            self.__operator = operation
            left = first if isinstance(first, Node) else Node(first)
            left.__parent = self
            self.__operand = [left]
            self.size = 1 + left.size
            self.valueCalc()

    def __isNumber(var: Union[str, float, int]) -> bool:
        """
        Checks whether a given value can be converted to a float.
    
        Args:
            var (str): The value to check.
    
        Returns:
            bool: True if the value can be parsed as a float, False otherwise.
        """
        try:
            float(var)
            return True
        except:
            return False
        
    def __isVariable(self) -> bool:
        """
        Checks whether the node represents a variable.
    
        Returns:
            bool: True if this node is a variable placeholder, False otherwise.
        """
        if not isinstance(self.__operand[0], str):
            return False
        return self.__operand[0].startswith(Node.__variableIndicator) and self.__operand[0].endswith(Node.__variableIndicator)
    
    def setOperation(self, newOperation: str) -> None:
        """
        Sets a new operation for the node and recalculates its value.
    
        Args:
            newOperation (str): The new operator ('+', '-', '*', or '/').
    
        Raises:
            AttributeError: If the provided operation is invalid.
        """
        if self.op == 2:
            if newOperation not in Node.__binaryOperations:
                raise AttributeError(f"Operation {newOperation} is not accepted!");
            self.__operator = newOperation
        elif self.op == 1:
            if newOperation not in Node.__unaryOperations:
                raise AttributeError(f"Operation {newOperation} is not accepted!");
            self.__operator = newOperation
        elif self.op == 0:
            raise AttributeError("This node is a leaf, it has no operation!")
        
        self.valueCalc()

    def setOperand(self, newOperand: Union['Node', str, int], operandPos: int = 0) -> None:
        """
        Sets a new operand for the node and recalculates its value.
    
        Args:
            newOperand (Union[Node, str, int]): The new operand or subtree.
            operandPos (int, optional): The position to update (0 for left, 1 for right). Defaults to 0.
    
        Raises:
            AttributeError: If the operand position is invalid or out of bounds.
        """
        if len(self.__operand) <= operandPos:
            raise AttributeError(f"Operand position: {operandPos} is not valid!")
        
        newNode = newOperand if isinstance(newOperand, Node) else Node(newOperand)
        newNode.__parent = self
        self.__operand[operandPos] = newNode.__operand[0]
        self.valueCalc()
        
    def __replaceNode(self, newNode: 'Node') -> None:
        """
        Replaces the current node with a new node.
    
        Args:
            newNode (Node): The new node to replace the current node.
    
        Raises:
            AttributeError: If the provided 'newNode' is not of type 'Node'.
        """
        if not isinstance(newNode, Node):
            raise AttributeError(f"Given newNode is of type {type(newNode)}, not of type Node!")
        
        self.__operator = newNode.__operator
        self.__operand = newNode.__operand
        self.size = newNode.size
        self.value = newNode.value
        self.__leaves = newNode.__leaves
        self.op = newNode.op
        
        for operand in self.__operand:
            if isinstance(operand, Node):
                operand.__parent = self 
        
        self.valueCalc()
    
    def valueCalc(self) -> None:
        """
        Recalculates the value of the expression represented by this node and updates the parent nodes.
    
        - Leaf nodes are evaluated directly based on their operand.
        - Internal nodes are evaluated based on their operator and the values of their children.
        - Handles division by zero by setting the value to infinity.
    
        Updates the following attributes:
            - value: The calculated value of the node.
            - __leaves: The list of leaf values in the subtree rooted at this node.
            - size: The size of the subtree.
        """
        
        tree = self
        while True:
            if tree.op == 0:
                tree.value = tree.__operand[0] if not tree.__isVariable() else None
                tree.__leaves = {tree.value} if tree.value is not None else {tree.__operand[0][11:-11]}
                tree.size = 1
            elif tree.op == 2:
                try:
                    leftVal = tree.__operand[0].value
                    rightVal = tree.__operand[1].value
                    tree.value = eval(f"{leftVal} {tree.__operator} {rightVal}") if leftVal != None and rightVal != None else None
                except(ZeroDivisionError, ValueError, OverflowError):
                    #raise AttributeError("Division with 0 undefined behavior!")
                    tree.value = None
                    
                tree.__leaves = tree.__operand[0].__leaves.union(tree.__operand[1].__leaves)
                tree.size = 1 + tree.__operand[0].size + tree.__operand[1].size
            elif tree.op == 1:
                try:
                    leftVal = tree.__operand[0].value
                    tree.value = Node.__mapUnary[tree.__operator](leftVal) if leftVal != None else None
                except(ZeroDivisionError, ValueError, OverflowError):
                    #raise AttributeError("Log function can not take 0, undefined behavior!")
                    tree.value = None

                tree.__leaves = tree.__operand[0].__leaves
                tree.size = 1 + tree.__operand[0].size
            
            if tree.__parent is None:
                break
            tree = tree.__parent

    def valueCalcVar(self, varVal = list[list[float]]) -> list[list[Union[str, float]]]:
        """
        Evaluates the expression tree for multiple variable assignments.
    
        Args:
            varVal (list[list[float]]): A table where:
                - The first row contains variable names (list[str]).
                - Each following row contains corresponding values for those variables.
    
        Returns:
            list[list[Union[str, float]]]: A table where:
                - The first row contains variable names plus "Value:".
                - Each subsequent row contains variable values and the computed result.
    
        Raises:
            AttributeError: If a variable in the tree is not present in 'varVal'.
        """
        
        for varName in self.getLeaves():
            if not Node.__isNumber(varName) and varName not in varVal[0]:
                raise AttributeError(f"There is no value for variable: {varName}")
        
        varMap = {}
        for i in range(len(varVal[0])):
            varMap[Node.__variableIndicator + varVal[0][i] + Node.__variableIndicator] = [v[i] for v in varVal[1:]]

        evaluated = [[str(k)[11:-11] for k in varMap.keys()]]
        evaluated[0].append('Value:')

        n = len(varVal)-1
        value = self.__rCalcVar(varMap, n) if self.value is None else [self.value] * n
        
        
        for i in range(1, len(varVal)):
            evaluated.append([v for v in varVal[i]])
            evaluated[i].append(value[i-1])

        return evaluated
                    
    def __rCalcVar(self, varVal: map, n: int) -> list:
        """
        Recursively evaluates the expression tree for variable assignments.
    
        Args:
            varVal (dict[str, list[float]]): A mapping from variable placeholders
                (with internal markers) to lists of values.
    
        Returns:
            list[float]: The computed values for each assignment, one per row in 'varVal'.
        """
        if self.value is not None:
            return [self.value] * n
        
        if self.op == 0:
            val = [num for num in varVal[self.__operand[0]]]
            return val
        elif self.op == 1:
            val = []
            for num in self.__operand[0].__rCalcVar(varVal, n):
                try:
                    compute = Node.__mapUnary[self.__operator](num) if num is not None else None
                    val.append(compute)
                except(ZeroDivisionError, ValueError, OverflowError):
                    val.append(None)
            return val
        elif self.op == 2:
            val = []
            leftVal = self.__operand[0].__rCalcVar(varVal, n) if self.__operand[0].value is None else self.__operand[0].value
            rightVal = self.__operand[1].__rCalcVar(varVal, n) if self.__operand[1].value is None else self.__operand[1].value
            
            leftVal = leftVal if isinstance(leftVal, list) else [leftVal] * n
            rightVal = rightVal if isinstance(rightVal, list) else [rightVal] * n
            
            for l, r in zip(leftVal, rightVal):
                try:
                    compute = eval(f"{l} {self.__operator} {r}") if l is not None and r is not None else None
                    val.append(compute)
                except(ZeroDivisionError, ValueError, OverflowError):
                    val.append(None)
            return val
    
    def subTree(self, pos: int, insertTree: 'Node' = None) -> 'Node':
        """
        Retrieves or replaces a subtree at a specified position in the binary tree.
    
        Args:
            pos (int): The position of the subtree (1-based index).
            insertTree (Node, optional): If provided, replaces the subtree at the position with this node.
    
        Returns:
            Node: The subtree at the specified position.
    
        Raises:
            AttributeError: If the position is out of bounds or 'insertTree' is not a valid node.
        """
        if insertTree != None and not isinstance(insertTree, Node):
            raise AttributeError(f"Given insertTree is of type {type(insertTree)}, not of type Node!")
        if pos > self.size:
            raise AttributeError(f"Goint out of bounds! Tree \"{str(self)}\" size {self.size}, trying to get subtree on node {pos}!")

        tree = self
        while pos != 1:
            if pos - tree.__operand[0].size <= 1:
                pos -= 1
                tree = tree.__operand[0]
            else:
                pos -= (tree.__operand[0].size + 1)
                tree = tree.__operand[1]
                
        if insertTree != None:
            tree.__replaceNode(insertTree)
            return tree
        return tree
        
    def getOperation(self) -> str:
        """
        Retrieves the operation of the node.
    
        Returns:
            str: The operator of the node.
    
        Raises:
            AttributeError: If the node has no operator (i.e., it is a leaf node).
        """
        if self.op == 0:
            raise AttributeError(f"This node has no operatino!")
        return self.__operator
    
    def getLeaves(self) -> list[Union[str, float]]:
        """
        Retrieves the list of leaf values in the subtree rooted at this node.
    
        Returns:
            list: A list of leaf values.
        """
        return [l for l in self.__leaves]

    def __str__(self) -> str:
        if self.op == 0:
            return f"{self.__operand[0] if not self.__isVariable() else self.__operand[0][11:-11]}"
        elif self.op == 1:
            return f"{self.__operator}({self.__operand[0]})"
        return f"({self.__operand[0]} {self.__operator} {self.__operand[1]})"

    def getSupportedOperations() -> (list[str], list[str]):
        """
        Returns the list of all supported operations for the Node class.
    
        Includes both binary and unary operations.
    
        Returns:
            (list[str], list[str]): First list of binary, second of unary operations.
        """
        return cp.deepcopy((Node.__binaryOperations, Node.__unaryOperations))
    
    def nodeInfo(self) -> str:
        leaves1 = self.getLeaves()
        leaves = []
        for l in leaves1:
            if not Node.__isNumber(l):
                leaves.append(l)
        return f"""{self}\n{f'Value: {self.value}' if self.value is not None else f'Value: NONE\nEquation has variables: {leaves}'}"""

    def __lt__(self, other) -> bool:
        a = float('inf') if self.value is None else self.value
        b = float('inf') if other.value is None else other.value
        return a < b