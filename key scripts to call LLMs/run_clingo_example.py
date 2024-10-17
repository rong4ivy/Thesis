import sys
from io import StringIO
import clingo
import dspy

class RunClingo(dspy.Module):
    class Signature(dspy.Signature):
        """Run Clingo and capture feedback."""
        asp_code = dspy.InputField(desc="ASP code to be evaluated")
        result = dspy.OutputField(desc="Result from Clingo")
        error = dspy.OutputField(desc="Error output from Clingo")

    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(self.Signature)

    def forward(self, asp_code):
        result, error = self.run_clingo(asp_code)
        return dspy.Prediction(result=result, error=error)

    def run_clingo(self, asp_program):
        results = []
        error_output = StringIO()

        try:
            original_stderr = sys.stderr
            sys.stderr = error_output

            ctl = clingo.Control()
            ctl.add("base", [], asp_program)
            ctl.ground([("base", [])])

            def on_model(model):
                for atom in model.symbols(shown=True):
                    if atom.name.startswith("query"):
                        if len(atom.arguments) > 0:
                            results.append(str(atom.arguments[0]))

            solve_result = ctl.solve(on_model=on_model)

            if solve_result.satisfiable:
                output = "Answer: " + ", ".join(results) if results else "satisfiable, but no query results"
            elif solve_result.unsatisfiable:
                output = "unsatisfiable"
            else:
                output = "Unknown"

        except Exception as e:
            output = f"Error: {str(e)}"
        finally:
            sys.stderr = original_stderr

        error_string = error_output.getvalue()
        error_output.close()

        return output, error_string

# Example usage
if __name__ == "__main__":
    
    asp_program = """
    
% Define blocks
block(aaa).
block(bbb).

% Define objects: object(distinctname, size, color, shape, block_located)
object(medium_yellow_square_1, medium, yellow, square, aaa).
object(medium_yellow_square_2, medium, yellow, square, aaa).
object(medium_blue_square, medium, blue, square, aaa).
object(medium_yellow_square_3, medium, yellow, square, aaa).
object(medium_black_unknown, medium, black, unknown, bbb). 

% Define spatial relations between blocks
is(bbb, over, aaa).
is(bbb, behind, aaa).
is(aaa, disconnected, bbb).
is(aaa, far_from, bbb).

% Define spatial relations between objects
is(medium_yellow_square_3, covered_by, aaa).
is(medium_yellow_square_3, touching, medium_yellow_square_2).
is(medium_yellow_square_1, touching, medium_blue_square).
is(medium_black_unknown, touching, bbb).

% Define general spatial relations
relation(over; behind; disconnected; far_from; covered_by; touching; in_front_of; left; right; above; below).

% Define inverse relations
inverse(over, under).
inverse(behind, in_front_of).
inverse(far_from, far_from).
inverse(disconnected, disconnected).
inverse(touching, touching).
inverse(covered_by, covers).
inverse(left, right).
inverse(above, below).

is(Y, R2, X) :- is(X, R1, Y), inverse(R1, R2).

% Define transitive relations
transitive(over; under; behind; in_front_of; left; right; above; below).

is(X, R, Z) :- is(X, R, Y), is(Y, R, Z), transitive(R), X != Z.

% Objects in a block are in the same position as the block relative to other blocks
is(Object, Relation, Block2) :-
    object(Object, _, _, _, Block1),
    block(Block2),
    Block1 != Block2,
    is(Block1, Relation, Block2).

% Query 1: Are all objects in front of the black object?
query1(yes):- 
    object(BlackObj, _, black, _, _),
    is(OtherObj, in_front_of, BlackObj) : object(OtherObj, _, _, _, _), OtherObj != BlackObj.
query1(no):- not query1(yes).

% Query 2: How many yellow squares are in block AAA?
query2(Count) :- 
    Count = #count{Obj : object(Obj, _, yellow, square, aaa)}.


% Query 3: Is there a blue object touching a yellow object?
query3(yes) :- 
    object(BlueObj, _, blue, _, _),
    object(YellowObj, _, yellow, _, _),
    is(BlueObj, touching, YellowObj).
query3(no) :- not query3(yes).

% Query 4: What is the relation between the black object and block AAA?
query4(Relation) :- 
    object(BlackObj, _, black, _, _),
    is(BlackObj, Relation, aaa).

% Query 5: Are all yellow squares in the same block?
query5(yes):- 
    1 = #count{Block : object(_, _, yellow, square, Block)}.
query5(no):- not query5(yes).

% Show directives
#show query1/1.
#show query2/1.
#show query3/1.
#show query4/1.
#show query5/1.

"""

runner = RunClingo()
result = runner.forward(asp_program)
print("Result:", result.result)
print("Error:", result.error)
    