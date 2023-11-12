class WCNF:
    """
    Helper class for reading and parsing WCNF files. Works only for weighted CNF without constraints.

    Attributes:
        weights (list): List of weights for each clause.
        clauses (list): List of clauses.
        nv (int): Number of variables.
    """

    def __init__(self, file_path: str):
        """
        Constructor for WCNF class.

        Args:
            file_path: Path to the WCNF file.
        """
        weights = []
        clauses = []

        with open(file_path, "r") as f:
            lines = f.readlines()
            lines = [line.strip().replace("\n", " ") for line in lines]
            lines = [
                line.strip().split(" ")[:-1]
                for line in lines
                if line != "" and line[0] != "c" and line[0] != "p"
            ]

            for line in lines:
                weight = int(line[0])
                clause = [int(literal) for literal in line[1:] if len(literal) > 0]
                weights.append(weight)
                clauses.append(clause)

            self.weights = weights
            self.clauses = clauses
            self.nv = max([abs(literal) for clause in clauses for literal in clause])
