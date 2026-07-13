import numpy as np

def getFIM(traces):
    """Get Fisher information matrix
    AKA matrix of overlap integrals"""
    return np.array([[np.sum(arr*arr2) for arr2 in traces] for arr in traces])



def getVariances(traces):
    """Get Variance estimates from FIM"""
    FIM=getFIM(traces)
    return np.diag(np.linalg.inv(FIM))
    
def getVariancesDueToOverlap(traces):
    """Calculate variance penalty due to non-orthogonality"""
    traces = [trace/np.sum(trace**2)**0.5 for trace in traces]
    FIM=getFIM(traces)
    return np.diag(np.linalg.inv(FIM))

def getMinimumVariance(traces):
    """ Returns eigenvals of the covariance matrix.
    This can be useful with e.g. xyz measurements to find most sensitive measurement directions 
    """
    fim=getFIM(traces)
    cov = np.linalg.inv(fim)
    return np.linalg.eigvals(cov)

def _getVarPerRateEstimates(traces, dt=1):
    FIMs = np.array([[np.cumsum(arr*arr2) for arr2 in traces] for arr in traces])
    #pdb.set_trace()
    #print(FIMs.T.shape)
    ests = np.array([np.diag(np.linalg.inv(FIM))*k*dt for k,FIM in enumerate(FIMs.T[:]) if k >3])
    #FIMs_cw = np.cumsum( np.ones(sigs.shape[1])* np.sqrt(np.sum(sigs[:,-1000:].mean(axis=1))**2))[None,None,:]
    #FIMs_cw = np.cumsum( np.ones(sigs.shape[1])* np.sqrt(np.sum(sigs[:,-2000:].mean(axis=1)**2, axis=0)))[None,None,:]

    #ests_cw = np.array([np.diag(np.linalg.inv(FIM))*k*dt for k,FIM in enumerate(FIMs_cw.T[:]) if k >3])
    
    return np.sqrt(ests)#, ests_cw

def _getVarPerRateEstimatesAB(sigsA,sigsB, dt=.005):
    """
    sigsA, sigsB: signature arrays that are trimmed simultaneously. E.g. the full signatures will be of the form hstack([sigsA[:,:k], sigsB[:,:k])
    """
    FIMs = np.array([[np.cumsum(arr1A*arr2A)+np.cumsum(arr1B*arr2B) for arr2A, arr2B in zip(sigsA,sigsB)] for arr1A, arr1B in zip(sigsA,sigsB)])
    #FIMs_B = np.array([[np.cumsum(arr1*arr2) for arr2 in sigs] for arr1 in sigsB])

    #pdb.set_trace()
    #print(FIMs.T.shape)
    ests = np.array([np.diag(np.linalg.inv(FIM))*(2*k*dt) for k,FIM in enumerate(FIMs.T[:]) if k >3])
    #ests = np.array([np.linalg.eigvals(np.linalg.inv(FIM))*(2*k*dt) for k,FIM in enumerate(FIMs.T[:]) if k >3])

    #FIMs_cw = np.cumsum( np.ones(sigs.shape[1])* np.sqrt(np.sum(sigs[:,-1000:].mean(axis=1))**2))[None,None,:]
    #FIMs_cw = np.cumsum( np.ones(sigs.shape[1])* np.sqrt(np.sum(sigs[:,-2000:].mean(axis=1)**2, axis=0)))[None,None,:]

    #ests_cw = np.array([np.diag(np.linalg.inv(FIM))*k*dt for k,FIM in enumerate(FIMs_cw.T[:]) if k >3])
    
    return np.sqrt(ests)#, ests_cw

def _extrapolate_cw_signature(sig_xy, frac_to_ave = 0.1):
    N = fract_to_ave*sig_xy.shape[-1]
    N_ave = int(N*frac_to_ave)
    
    return (np.ones(N)* np.sqrt(np.sum(sig_xy[..., -Nave:].mean(axis=-1)**2, axis=-1)) )[None,:]