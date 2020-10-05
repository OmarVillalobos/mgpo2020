from pgmpy.factors.discrete import DiscreteFactor
from numpy import prod

def variable_elimination(factors, query_variables, evidence=None, elimination_order=None):
    """
    This function takes as inputs:
     - The set of factors $\bar{\Phi}$ that model the problem.
     - The variables that won't be eliminated Y (query variables).
     - The evidence (E=e).
     - The elimination order.
     
    And returns the inferred probability P(Y|E=e).
    
    :param list[DiscreteFactor] factors: List of factors that model the problem.
    :param list[str] query_variables: Query variables.
    :param dict{str: int} evidence: Evidence in the form of a dictionary. For example evidence={'D': 2, 'E':0}
                                    means that D=d² and E=e⁰.
    :param list[str] elimination_order: Specification of the order in which the variables will be eliminated.
    :return: DiscreteFactor corresponding the inferred probability.
    """
    # --------------------------------------------- Parameter check ---------------------------------------------
    if not isinstance(factors, list) and not factors:
        raise ValueError(f"The parameter factors: {factors} must be a nonempty list of DiscreteFactor objects.")
    if not isinstance(query_variables, list) and not query_variables:
        raise ValueError(f"The parameter query_variables: {query_variables} must be a nonempty list of str objects.")
    if evidence is not None and (not isinstance(evidence, dict) and not evidence):
        raise ValueError(f"The parameter evidence: {evidence} must be a nonempty dict.")
    if elimination_order is not None and (not isinstance(elimination_order, list) and not elimination_order):
        raise ValueError(f"The parameter elimination_order: {elimination_order} must be a nonempty list of str objects.")
    # --------------------------------------------- End parameter check -----------------------------------------
    
    # Initial parameters
    # Number of factors
    m = len(factors)
    # Get variables
    variables = []
    for i in range(m):
        variables.extend(factors[i].variables)
    variables = list(set(variables))

    #Number of variables
    n = len(variables)

    # 1. If evidence is not None, we must reduce all the factors according to the evidence
    if evidence is not None:
        evidence_variables = list(evidence.keys())
        # For each factor
        for i in range(m):
            # Find intersection of variables between evidence and factor scope
            intersection = set(evidence_variables).intersection(set(factors[i].variables))
            # If intersection is not empty, we must reduce this factor
            if intersection:
                ev = {var: evidence[var] for var in intersection if var in evidence}
                factors[i] = factors[i].reduce(ev.items(), inplace=False)

    # Variables to eliminate
    variables_to_eliminate = set(variables).difference(set(query_variables))
    if evidence is not None:
        variables_to_eliminate = variables_to_eliminate.difference(set(evidence_variables))
    variables_to_eliminate = list(variables_to_eliminate)

    # If elimination_order is not None, we must check if the variables in elimination_order are right.
    # If the variables in elimination_order are right, then they should be set as variables_to_eliminate.
    if elimination_order is not None:
        if set(elimination_order).difference(set(variables_to_eliminate)):
            raise ValueError(f"The parameter elimination_order: {elimination_order} does not contain the right variables.")
        else:
            variables_to_eliminate = elimination_order

    # 2. Eliminate Var-Z
    factors_update = factors.copy()

    for var in range(len(variables_to_eliminate)):
        # ---------------------------- Your code goes here! ----------------------------------
        # Determine the set of factors that involve var
        pos = []
        not_pos = []
        for f in range(len(factors_update)):
            if variables_to_eliminate[var] in factors_update[f].variables:
                pos.append(f)
            else:
                not_pos.append(f)
        involved_f = [factors_update[i] for i in pos]
        not_involved_f = [factors_update[i] for i in not_pos]
        # Compute the product of these factors
        product = prod(involved_f)
        # Marginalize var
        product.marginalize(variables = [variables_to_eliminate[var]])
        # Overwrite factors_update
        not_involved_f.append(product)
        factors_update = not_involved_f.copy()
        # ------------------------------------------------------------------------------------
    
    # 3. Multiply the remaining factors
    # ---------------------------- Your code goes here! ----------------------------------
    return (print(prod(factors_update).normalize()))
    # ------------------------------------------------------------------------------------

# Test
# Define factors
phi_L = DiscreteFactor(variables=['L'],
                       cardinality=[2],
                       values=[0.4, 0.6])
phi_Q = DiscreteFactor(variables=['Q'],
                       cardinality=[3],
                       values=[0.2, 0.5, 0.3])
phi_C = DiscreteFactor(variables=['C', 'L', 'Q'],
                       cardinality=[2, 2, 3],
                       values=[0.95, 0.4, 0.4, 0.9, 0.4, 0.2,
                               0.05, 0.6, 0.6, 0.1, 0.6, 0.8])
phi_N = DiscreteFactor(variables=['N', 'L', 'C'],
                       cardinality=[2, 2, 2],
                       values=[0.4, 0.9, 0.2, 0.4,
                               0.6, 0.1, 0.8, 0.6])

                               # Joint probability
P_LQCN = phi_L * phi_Q * phi_C * phi_N

# P(N)
P_QCN = P_LQCN.marginalize(variables=['L'], inplace=False)
# print(P_QCN)

P_QN = P_QCN.marginalize(variables = ['C'], inplace=False)
# print(P_QN)

# testing
variable_elimination([phi_L,phi_Q,phi_C,phi_N],['N'], {'L':0}, ['Q','C'])