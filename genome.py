from node import Node
import copy as cp
import random
import math as m

# n is size of tree
    #__makeGene : O(n)
    # __generate : O(n)
    # getFitness : O(n)
    # isValid : O(n^2) removal of elements in list is O(N) where N is size of list
    # mutate() : O(n log n)  
    # _mutateNode() : O(n log n)

class Genome:
    """
    Represents a genome with a specific target fitness goal and a list of numerical values. 
    The genome is structured as a binary expression tree and evaluated based on its 
    closeness to the target goal using mathematical operations.

    Attributes:
        goal (int): The target value that the genome attempts to approximate.
        numbers (list): A list of integers used as leaf nodes in the gene's binary tree structure.
        chance (float): The probability of mutation occurring in the gene structure.
        gene (Node): The root of the binary tree representing the genome's expression.
        
    Methods:
        getFitness(): Calculates the fitness of the genome based on its approximation to the target goal.
        isValid(): Validates if the genome's expression is integer-valued and uses only the specified numbers.
        mutate(): Applies a random mutation to the genome, potentially altering nodes and operations in the expression tree.

    Private methods:
        __makeGene(): Constructs the initial binary expression tree with random operations and operands.
        __mutateNode(): Internal helper for `mutate()` that performs single-node mutations within the tree, 
                        altering operands or operators.
        __str__(): Returns a string representation of the genome's expression.
        __lt__(other): Defines comparison based on genome fitness and expression value. 
    """
    
    __numbers = [m.pi, m.e]
    
    def __init__(self, test: list[list[float]], variables: list[str], treeDepth: int, chance: float = 0.05):
        """
        Initializes a Genome instance with a matrix where every row containg all values for variables and target goal for respective row,
        list of variables, and a mutation chance.
        
        Args:
            goal (int): The target integer value for the fitness function.
            numbers (list): A list of integers used in constructing the binary tree.
            chance (float, optional): The probability of mutation in the genome; default is 0.05.
        
        Raises:
            AttributeError: If the `numbers` list is empty.
        """
        if 2**(treeDepth-1) < len(variables):
            raise AttributeError(f"Tree needs to have variables! Tree depth is set : {treeDepth} but there is {len(variables)} variables!")
        
        if len(variables) == 0:
            raise AttributeError("List needs to have at least one variable!")

        self.test = test
        self.variables = variables
        self.chance = chance
        self.treeDepth = treeDepth

        self.numbers = [1e-10 + random.random() if random.random() > self.chance 
                        else _ for _ in range(self.treeDepth)] + Genome.__numbers

        self.gene = self.__makeGene()
        self.geneError = None
    
    def __makeGene(self) -> 'Node':
        """
        Constructs the initial binary expression tree for the genome.
    
        If the `numbers` list contains only one value, creates a single-node tree.
        Otherwise, randomly selects operands and constructs a binary tree with random operations.
    
        Returns:
            Node: The root of the binary expression tree representing the genome.
        """
        if len(self.variables) == 1 and random.random() < self.chance:
            return Node(self.variables[0])

        numOfconsts = random.randint(1, len(self.variables))
        numbers = random.choices(self.numbers, k = numOfconsts)
        chosenOperands = self.variables + numbers
        
        return self.__generate(chosenOperands)

    def __generate(self, givenList: list) -> 'Node':
        """
        Recursively generates a binary tree from a list of operands with random operations.
    
        Args:
            givenList (list): A list of operands to construct the binary tree.
    
        Returns:
            Node: The root of the binary tree constructed from the given list.
        """
        opList = Node.getSupportedOperations()
        
        if len(givenList) == 1:
            if random.random() < self.chance:
                op = random.choice(opList[1])
                return Node(givenList[0], op)
            return Node(givenList[0])

        if len(givenList) == 2:
            if random.random() < self.chance:
                op = random.choice(opList[1])
                return Node(givenList[0], op)
                
            op = random.choice(opList[0])
            return Node(givenList[0], op, givenList[1])
        
        if len(givenList) > 2:
            opBU = random.choice([0, 1])
            if opBU == 0:
                op = random.choice(opList[opBU])
                index = random.randint(1, len(givenList) - 1) 
                left = self.__generate(givenList[:index]) 
                right = self.__generate(givenList[index:])  
                return Node(left, op, right)
            elif opBU == 1:
                op = random.choice(opList[opBU])
                left = self.__generate(givenList)
                return Node(left, op)
         
        
    def getFitness(self) -> float:
        """
        Calculates the fitness of the genome based on its closeness to the target goal.
    
        The fitness is scaled between 0 and 100, where higher values represent better approximations.
        If the genome's expression is invalid, the fitness is 0.
    
        Returns:
            float: The fitness score of the genome.
        """
        if not self.isValid():
            self.geneError = float('inf')
            return self.geneError

        self.geneError = 0
        values = self.gene.valueCalcVar(self.test)
        
        for i in range(1, len(values)):
            if values[i][-1] is None or isinstance(values[i][-1], complex):
                self.geneError = float('inf')
                return self.geneError
            try:
                self.geneError += (values[i][-1] - self.test[i][-1])**2   
            except(OverflowError):
                self.geneError = float('inf')
        self.geneError /= (len(self.test)-1)

        return self.geneError

    def isValid(self):
        """
        Validates whether the genome's expression meets the required constraints.
    
        Checks that:
        - The expression evaluates to an integer.
        - The size of the binary tree does not exceed the maximum allowable size.
        - The expression uses only the specified numbers.
    
        Returns:
            bool: True if the genome is valid, False otherwise.
        """
        if self.gene.size > 2**(self.treeDepth) - 1:
            return False
            
        try:
            leaves = cp.copy(self.gene.getLeaves())
            for v in self.variables:
                leaves.remove(v)
            return True
        except (ValueError):
            return False

    def reEvalNumTerms(self):
        leaves = self.gene.getLeaves()
        for l in leaves:
            try: 
                n = float(l)
                if n not in self.numbers:
                    numbers.append(n)
            except:
                continue
                
    def mutate(self):
        """
        Applies mutation to the genome, modifying its expression tree.
    
        Mutations include:
        - Adding a new subtree with a random operand and operation.
        - Replacing a subtree with a new randomly generated tree.
        - Mutating individual nodes in the expression tree by altering their operands or operations.
    
        The mutation occurs probabilistically based on the mutation chance.
        """
        
        if random.random() < self.chance and self.gene.size > 1:
            self.__growMutateVN()

        if random.random() < self.chance and self.gene.size > 1:
            self.__growMutateN()
        
        self.__mutateNode()

    def __growMutateVN(self) -> None:
        numOfconsts = random.randint(1, len(self.variables))
        numbers = random.choices(self.numbers, k = numOfconsts)
        chosenOperands = self.variables + numbers
        nodeIn = self.__generate(chosenOperands)
        
        num = random.randint(1, self.gene.size)
        self.gene.subTree(num, nodeIn)

    def __growMutateN(self) -> None:
        numOfconsts = random.randint(1, len(self.numbers))
        numbers = random.choices(self.numbers, k = numOfconsts)
        nodeIn = self.__generate(numbers)

        num = random.randint(2, self.gene.size)
        self.gene.subTree(num, nodeIn)
    
    def __mutateNode(self) -> None:
        """
        Mutates individual nodes within the binary expression tree.
    
        For each node, randomly decides whether to mutate it. If the node is:
        - A leaf node: Replaces its operand with a random value from the list of allowed numbers.
        - An internal node: Replaces its operation with a randomly chosen operation.
    
        Mutation is probabilistic and depends on the mutation chance.
        """
        if self.gene.size < 2:
            return
        
        for i in range(1, self.gene.size):
            if random.random() < self.chance:
                tree = self.gene.subTree(i)
                if tree.op == 0:
                    numbers = random.choices(self.numbers, k = len(self.numbers))
                    varOrNum = random.choices(population=[self.variables, self.numbers], weights = [8, 1], k=1)[0]
                    tree.setOperand(random.choice(varOrNum))
                elif tree.op == 1:
                    operations = Node.getSupportedOperations()[1]
                    operations.remove(tree.getOperation())
                    tree.setOperation(random.choice(operations))
                elif tree.op == 2:
                    operations = Node.getSupportedOperations()[0]
                    operations.remove(tree.getOperation())
                    tree.setOperation(random.choice(operations))
                break
    
    def __str__(self):
        return str(self.gene)

    def __lt__(self, other):
        a = self.geneError if self.geneError is not None else self.getFitness()
        b = other.geneError if other.geneError is not None else other.getFitness()
        return a < b