
# ## to build the near-term and next term options data
# 
# intended to find the two available option expirations about a target maturity say 30, 60, 90, 120, 180 and 360. so we can interpolate a constant-maturity value
# 

import numpy as np

def bracket_expirations(available_dte: np.ndarray, target: int)-> tuple[int,int]:
    """
    available_dte: should be a sorted array of days-to-expiration on offer for a 
    given trading day. (after converting to date time)
    target: target constant maturity in calender days

    returns: s1-> upper bound and s2-> lower bount where s1<=target<=s2,
    using the closest available expiration on each side. if the target exactly matches
    an available expiration, it returns (S,S)
    """
    dtes = np.asarray(sorted(set(available_dte)))
    # takes whatever days-to-expiration on that trading day and turns them into 
    # a sorted array of unique values
    if len(dtes) ==0:
        return None
    # if there are no expirations at all, return none
    if target in dtes:
        return int(target), int(target)
    # this signals that no interpolation is needed, just use this single maturity with full weight

    below = dtes[dtes < target]
    above = dtes[dtes > target]

    if len(below) ==0 or len(above) ==0:
        return None #if the required target falls outside the range of whats actually
    #listed that day, in that case there's no valid bracket, so return none
    
    #pick the nearest neighbours.
    s1 = int(below.max())
    s2 = int(above.min())
    return s1,s2


