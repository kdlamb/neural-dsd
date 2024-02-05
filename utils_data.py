import numpy as np
from sklearn.preprocessing import MinMaxScaler


def logrescale(inputdata,momflag=[0,3,6,9],minmax=True):
    # converts the moments to indices
    scaler = MinMaxScaler()
    momidx = np.array(momflag,dtype=int)/3
    scaledinput = np.zeros(inputdata.shape)
    maskabledata = np.zeros((inputdata.shape[0],len(momflag)))
    scaledinputcol = np.zeros(inputdata.shape[0])

    for i in range(0,momidx.shape[0]):
        data = inputdata[:,int(momidx[i])]
        mask = np.argwhere(data>0)

        maskabledata[mask,i] = 1
        #print(momidx[i],mask.shape)
        scaledinputcol[mask] = np.log(data[mask])

        scaledinput[:,int(momidx[i])]=scaledinputcol[:]

    mask = np.all(maskabledata>0,axis=1)
    #print("all mask",mask.shape)
    if minmax==True:
        scaledinput = scaler.fit_transform(scaledinput)
    return scaledinput,mask

def twocatdata(ds):
    cloud_rates = ['cauto','caccr','ccoal']
    rain_rates = ['auto','accr','coal']
    ncases,nmoms,ntimes,nalts = ds['ccoal'].shape
    nrates = len(cloud_rates)

    t0 = 1
    ntimes-=1

    #cliq_mom(case, cloud_moment, time, altitude)
    #rain_mom(case, rain_moment, time, altitude)
    #cauto(case, cloud_moment, time, altitude)
    #caccr(case, cloud_moment, time, altitude)
    #ccoal(case, cloud_moment, time, altitude)
    #auto(case, rain_moment, time, altitude)
    #accr(case, rain_moment, time, altitude)
    #coal(case, rain_moment, time, altitude)

    twocatrates = np.zeros((ncases*(ntimes-1)*nalts,nmoms,nrates,2))
    twocatmoments = np.zeros((ncases*(ntimes-1)*nalts,nmoms,nrates,2))

    cloudmoments = np.moveaxis(ds['cliq_mom'],1,3)
    rainmoments = np.moveaxis(ds['rain_mom'],1,3)
    cloudmoments = cloudmoments[:,t0:-1,:,:].reshape(ncases*(ntimes-1)*nalts,nmoms)
    rainmoments = rainmoments[:,t0:-1,:,:].reshape(ncases*(ntimes-1)*nalts,nmoms)

    for i in range(0,nrates):
        cloud_rate = cloud_rates[i]
        rain_rate = rain_rates[i]
        #print(cloud_rate,rain_rate)
        cloudrate = np.moveaxis(ds[cloud_rate],1,3)
        cloudrate = cloudrate[:,t0:-1,:,:].reshape(ncases*(ntimes-1)*nalts,nmoms)
        rainrate = np.moveaxis(ds[rain_rate],1,3)
        rainrate = rainrate[:,t0:-1,:,:].reshape(ncases*(ntimes-1)*nalts,nmoms)

        for j in range(0,nmoms):
            twocatrates[:,j,i,0]=cloudrate[:,j]
            twocatrates[:,j,i,1]=rainrate[:,j]
            twocatmoments[:,j,i,0]=cloudmoments[:,j]
            twocatmoments[:,j,i,1]=rainmoments[:,j]
    
    return twocatmoments,twocatrates
def masktwocat(M,dMdt,thresh=1e-15):
    #two cat: 
    #nsamp, nrates,nmoms,ncat (cloud, rain)
    # rates 
    # cloud_rates = ['cauto','caccr','ccoal']
    # rain_rates = ['auto','accr','coal']
    
    nsamp,nmoms,nrates,ncat = M.shape
    mask = np.zeros((nsamp,nmoms,nrates,ncat))

    # index of 0th moment
    M0 = 0
    # index of 3rd moment
    M3 = 1
    
    for c in range(0,nmoms):
        for k in range(0,ncat):
            threshold = np.max(np.abs(M[:,0,c,k]))*thresh #1e-5
            mask[np.argwhere(M[:,0,c,k]<threshold),c] = 1

    # if the cloud is too low (should I also see if there's no rain?)
    fewdropsflag = np.argwhere(M[:,M0,0,0]<1e-2)
    lowmassflag = np.argwhere(M[:,M3,0,0]<1e-15)


    maskall = np.any(mask,axis=(1,2,3))
    print(maskall.shape)

    cloudmask2 = np.zeros(M.shape[0])
    cloudmask2[fewdropsflag]=1
    print(" Few drops: ", np.sum(cloudmask2))
    cloudmask2[lowmassflag]=1
    print("+ Low mass: ",np.sum(cloudmask2))
    cloudmask2[maskall]=1
    print("+ Really Low Moments: ",np.sum(cloudmask2))

    M[np.argwhere(cloudmask2==1),:,:,:]=0
    dMdt[np.argwhere(cloudmask2==1),:,:,:]=0

    return M[:,:,0,:],dMdt,cloudmask2

def rescaletwocat(M):
    # index of collision coalescence in proc_rates

    logmoments,maskmoments = logrescale(M,momflag=[0,3,6])

    return logmoments,maskmoments


def rescalesinglecat(M,dMdt,rate):
    # index of collision coalescence in proc_rates
    nact=-1
    ncc = 1
    if rate==ncc:
        logtar,masktar = logrescale(dMdt,momflag=[0,6,9])
    elif rate==nact:
        logtar,masktar = logrescale(dMdt,momflag=[0])
    else:
        logtar,masktar = logrescale(dMdt)
    logmoments,maskmoments = logrescale(M)

    maskall = np.logical_and(masktar,maskmoments)

    #print(logtar[maskall].shape)
    #print(logmoments[maskall].shape)

    return logmoments,logtar,maskall
def singlecatdata(ds):
    # this returns the moments and rates for collision-coalescence for the single category case for t0

    proc_rates = ['cevap','ccoal','csed']
    nrates = len(proc_rates)
    momidxs = [2,3,4,5]
    ncases, nmoms, ntimes, nalts = ds['ccoal'].shape
    nmoms = len(momidxs)

    t0 = 1
    ntimes-=1

    singlecatrates = np.zeros((ncases*(ntimes-1)*nalts,nmoms,nrates))
    singlecatmoments = np.zeros((ncases*(ntimes-1)*nalts,nmoms,nrates))

    singlecatmoment = np.moveaxis(ds['cliq_mom'],1,3)
    singlecatmoment = singlecatmoment[:,t0:-1,:,momidxs].reshape(ncases*(ntimes-1)*nalts,nmoms)
    
    for i in range(0,nrates):
        proc_rate = proc_rates[i]
        singlecatrate = np.moveaxis(ds[proc_rate],1,3)
        singlecatrate = singlecatrate[:,t0:-1,:,momidxs].reshape(ncases*(ntimes-1)*nalts,nmoms)

        singlecatrates[:,:,i]=singlecatrate
        singlecatmoments[:,:,i]=singlecatmoment

        # correct the moments for the order the process rates are calculated in Tau
        singlecatmoment = singlecatmoment-singlecatrate

    return singlecatmoments,singlecatrates
# Two category data - collision coalescence regimes
    
def twocatdata(ds):
    cloud_rates = ['cauto','caccr','ccoal']
    rain_rates = ['auto','accr','coal']
    ncases,nmoms,ntimes,nalts = ds['ccoal'].shape
    nrates = len(cloud_rates)

    t0 = 1
    ntimes-=1

    #cliq_mom(case, cloud_moment, time, altitude)
    #rain_mom(case, rain_moment, time, altitude)
    #cauto(case, cloud_moment, time, altitude)
    #caccr(case, cloud_moment, time, altitude)
    #ccoal(case, cloud_moment, time, altitude)
    #auto(case, rain_moment, time, altitude)
    #accr(case, rain_moment, time, altitude)
    #coal(case, rain_moment, time, altitude)

    twocatrates = np.zeros((ncases*(ntimes-1)*nalts,nmoms,nrates,2))
    twocatmoments = np.zeros((ncases*(ntimes-1)*nalts,nmoms,nrates,2))

    cloudmoments = np.moveaxis(ds['cliq_mom'],1,3)
    rainmoments = np.moveaxis(ds['rain_mom'],1,3)
    cloudmoments = cloudmoments[:,t0:-1,:,:].reshape(ncases*(ntimes-1)*nalts,nmoms)
    rainmoments = rainmoments[:,t0:-1,:,:].reshape(ncases*(ntimes-1)*nalts,nmoms)

    for i in range(0,nrates):
        cloud_rate = cloud_rates[i]
        rain_rate = rain_rates[i]
        #print(cloud_rate,rain_rate)
        cloudrate = np.moveaxis(ds[cloud_rate],1,3)
        cloudrate = cloudrate[:,t0:-1,:,:].reshape(ncases*(ntimes-1)*nalts,nmoms)
        rainrate = np.moveaxis(ds[rain_rate],1,3)
        rainrate = rainrate[:,t0:-1,:,:].reshape(ncases*(ntimes-1)*nalts,nmoms)

        for j in range(0,nmoms):
            twocatrates[:,j,i,0]=cloudrate[:,j]
            twocatrates[:,j,i,1]=rainrate[:,j]
            twocatmoments[:,j,i,0]=cloudmoments[:,j]
            twocatmoments[:,j,i,1]=rainmoments[:,j]
    
    return twocatmoments,twocatrates
def masksinglecat(M,dMdt,thresh=1e-15):
    nsamp,nmoms,nrates = M.shape
    mask = np.zeros((nsamp,nmoms))
    dMdtsign = np.zeros((nsamp,nmoms,nrates))

    # index of collision coalescence in proc_rates
    ncc = 1
    # index of 0th moment
    M0 = 0
    # index of 3rd moment
    M3 = 1
    dMdt[:,M3,ncc] = dMdt[:,M3,ncc]*0.0


    for j in range(0,nrates):
        for c in range(0,nmoms):
            if c>3 and j==ncc:
                mask[np.argwhere(dMdt[:,c,j]<0),c] = 1
                dMdt[np.argwhere(dMdt[:,c,j]<0),c,j] = 0
            dMdtsign[:,c,j]=dMdt[:,c,j]/np.abs(dMdt[:,c,j])
            dMdt[:,c,j] = np.abs(dMdt[:,c,j])

            threshold = np.max(np.abs(M[:,c,j]))*thresh #1e-5
            mask[np.argwhere(M[:,c]<threshold),c] = 1

        fewdropsflag = np.argwhere(M[:,M0,j]<1e-2)
        lowmassflag = np.argwhere(M[:,M3,j]<1e-15)

    maskall = np.any(mask,axis=1)
    #print(maskall.shape)

    cloudmask2 = np.zeros(M.shape[0])
    cloudmask2[fewdropsflag]=1
    print(" Few drops: ", np.sum(cloudmask2))
    cloudmask2[lowmassflag]=1
    print("+ Low mass: ",np.sum(cloudmask2))
    cloudmask2[maskall]=1
    print("+ Really Low Moments and neg. process rates: ",np.sum(cloudmask2))

    M[np.argwhere(cloudmask2==1),:,:]=0
    dMdt[np.argwhere(cloudmask2==1),:,:]=0
    dMdtsign[np.isnan(dMdtsign)]=0

    return M,dMdt,dMdtsign,cloudmask2
def m3_to_q():
    rhow = 1000.0
    return np.pi/6*rhow
def q_to_m3():
    return 1.0/m3_to_q()
def bindatanormed(ds,normalized=True):
    # this returns the number and mass size distributions at this time step (t0) and the next time step (t1)
    # If normalized == True, normalized to add up to one
    # and the magnitude relative to the max in the data set is returned as a value between 0 and 1

    ncases, ntimes, nalts, nbins = ds['dsd_number'].shape
    #print(ncases, ntimes, nalts, nbins)
    #dsd_number(case, time, altitude, bin_mass)
    t0 = 1
    ntimes-=1

    #this factor converts massbins to M3
    rhow = 1000.0
    factor = np.pi/6*rhow

    nbin0 = ds['dsd_number'][:,t0:-1,:,:].reshape(ncases*(ntimes-1)*nalts,nbins)
    mbin0 = ds['dsd_mass'][:,t0:-1,:,:].reshape(ncases*(ntimes-1)*nalts,nbins)/factor

    #dsd_mass(case, time, altitude, bin_mass)
    nbin1 = ds['dsd_number'][:,(t0+1):,:,:].reshape(ncases*(ntimes-1)*nalts,nbins)
    mbin1 = ds['dsd_mass'][:,(t0+1):,:,:].reshape(ncases*(ntimes-1)*nalts,nbins)/factor

    nbin1coal = nbin0+ds['dsd_number_coal'][:,t0:-1,:,:].reshape(ncases*(ntimes-1)*nalts,nbins)
    nbin1condevap = nbin0+ds['dsd_number_condevap'][:,t0:-1,:,:].reshape(ncases*(ntimes-1)*nalts,nbins)

    mbin1coal = mbin0+ds['dsd_mass_coal'][:,t0:-1,:,:].reshape(ncases*(ntimes-1)*nalts,nbins)/factor
    mbin1condevap = mbin0+ds['dsd_mass_condevap'][:,t0:-1,:,:].reshape(ncases*(ntimes-1)*nalts,nbins)/factor
            
    bin0 = np.zeros((ncases*(ntimes-1)*nalts,2,nbins))
    bin1coal = np.zeros((ncases*(ntimes-1)*nalts,2,nbins))
    bin1condevap = np.zeros((ncases*(ntimes-1)*nalts,2,nbins))
    binmag = np.zeros((ncases*(ntimes-1)*nalts,6))

    
    nbinsum = np.sum(nbin0,axis=1)
    mbinsum = np.sum(mbin0,axis=1)
    
    nbin1coalsum = np.sum(nbin1coal,axis=1)
    mbin1coalsum = np.sum(mbin1coal,axis=1)
    
    nbin1condevapsum = np.sum(nbin1condevap,axis=1)
    mbin1condevapsum = np.sum(mbin1condevap,axis=1)

    print(nbin0.shape)
    print(nbinsum.shape)
    if normalized:
        print("normalized")
        for k in range(0,nbins):
            nbin0[:,k]=nbin0[:,k]/nbinsum[:]
            mbin0[:,k]=mbin0[:,k]/mbinsum[:]

            nbin1coal[:,k]=nbin1coal[:,k]/nbinsum[:] #nbin1coalsum[:]
            mbin1coal[:,k]=mbin1coal[:,k]/mbinsum[:] #mbin1coalsum[:]

            nbin1condevap[:,k]=nbin1condevap[:,k]/nbinsum[:] #nbin1condevapsum[:]
            mbin1condevap[:,k]=mbin1condevap[:,k]/mbinsum[:] #mbin1condevapsum[:]
    
    
    bin0[:,0,:]=nbin0
    bin0[:,1,:]=mbin0

    bin1coal[:,0,:]=nbin1coal
    bin1coal[:,1,:]=mbin1coal

    bin1condevap[:,0,:]=nbin1condevap
    bin1condevap[:,1,:]=mbin1condevap

    binmag[:,0]=nbinsum
    binmag[:,1]=mbinsum
    binmag[:,2]=nbin1coalsum
    binmag[:,3]=mbin1coalsum
    binmag[:,4]=nbin1condevapsum
    binmag[:,5]=mbin1condevapsum
    
    nmoms = 4
    momscales = np.zeros(nmoms)

    for i in range(0,nmoms):
        Mx = ds['cliq_mom'][:,2+i,t0:,:].reshape(ncases*(ntimes)*nalts)
        momscales[i] = np.max(Mx)
    
    return bin0,bin1coal,bin1condevap,binmag,momscales

def bindata(ds,rescaled=True):
    # this returns the number and mass size distributions at this time step (t0) and the next time step (t1)
    # If rescaled == True they are normalized by the sum over the bins at t0
    # and the magnitude relative to the max in the data set is returned as a value between 0 and 1

    ncases, ntimes, nalts, nbins = ds['dsd_number'].shape
    #print(ncases, ntimes, nalts, nbins)
    #dsd_number(case, time, altitude, bin_mass)
    t0 = 1
    ntimes-=1

    #this factor converts massbins to M3
    rhow = 1000.0
    factor = np.pi/6*rhow

    nbin0 = ds['dsd_number'][:,t0:-1,:,:].reshape(ncases*(ntimes-1)*nalts,nbins)
    mbin0 = ds['dsd_mass'][:,t0:-1,:,:].reshape(ncases*(ntimes-1)*nalts,nbins)/factor

    #dsd_mass(case, time, altitude, bin_mass)
    nbin1 = ds['dsd_number'][:,(t0+1):,:,:].reshape(ncases*(ntimes-1)*nalts,nbins)
    mbin1 = ds['dsd_mass'][:,(t0+1):,:,:].reshape(ncases*(ntimes-1)*nalts,nbins)/factor

    nbin1coal = nbin0+ds['dsd_number_coal'][:,t0:-1,:,:].reshape(ncases*(ntimes-1)*nalts,nbins)
    nbin1condevap = nbin0+ds['dsd_number_condevap'][:,t0:-1,:,:].reshape(ncases*(ntimes-1)*nalts,nbins)

    mbin1coal = mbin0+ds['dsd_mass_coal'][:,t0:-1,:,:].reshape(ncases*(ntimes-1)*nalts,nbins)/factor
    mbin1condevap = mbin0+ds['dsd_mass_condevap'][:,t0:-1,:,:].reshape(ncases*(ntimes-1)*nalts,nbins)/factor

    nmoms = 4
    momscales = np.zeros(nmoms)

    for i in range(0,nmoms):
        Mx = ds['cliq_mom'][:,2+i,t0:,:].reshape(ncases*(ntimes)*nalts)
        momscales[i] = np.max(Mx)

    #print(Mx.shape,momscales)

    if rescaled:
        # These lines scale the bin values by the max. of M0 or M3
        for k in range(0,nbins):
            nbin0[:,k]=nbin0[:,k]/momscales[0]
            mbin0[:,k]=mbin0[:,k]/momscales[1]

            nbin1coal[:,k]=nbin1coal[:,k]/momscales[0]
            mbin1coal[:,k]=mbin1coal[:,k]/momscales[1]

            nbin1condevap[:,k]=nbin1condevap[:,k]/momscales[0]
            mbin1condevap[:,k]=mbin1condevap[:,k]/momscales[1]

    bin0 = np.zeros((ncases*(ntimes-1)*nalts,2,nbins))
    bin1coal = np.zeros((ncases*(ntimes-1)*nalts,2,nbins))
    bin1condevap = np.zeros((ncases*(ntimes-1)*nalts,2,nbins))
    binmag = np.zeros((ncases*(ntimes-1)*nalts,2))

    bin0[:,0,:]=nbin0
    bin0[:,1,:]=mbin0

    bin1coal[:,0,:]=nbin1coal
    bin1coal[:,1,:]=mbin1coal

    bin1condevap[:,0,:]=nbin1condevap
    bin1condevap[:,1,:]=mbin1condevap

    nbinsum = np.sum(nbin0,axis=1)
    mbinsum = np.sum(mbin0,axis=1)

    binmag[:,0]=nbinsum
    binmag[:,1]=mbinsum

    return bin0,bin1coal,bin1condevap,binmag,momscales
def flaggeddatanormed(ds1,ds2,normalized=True,twocat=False):
    # get rid of bad data and return the normalized data sets
    
    ### Single category data ###
    # Load the single cat data (Sample, moment, rate)
    
    ncases, ntimes, nalts, nbins = ds1['dsd_number'].shape
    ntimes-=1
    
    
    if twocat==False:
        M,dMdt = singlecatdata(ds1)
    
        # Flag data below thresholds and with no clouds
        M,dMdt,dMdtsign,mask0 = masksinglecat(M,dMdt)
        
        #print(momscales)
        nmoms = 4
        nrates = 3
        t0 = 1
        
        momscales = np.zeros(nmoms)
        
        for i in range(0,nmoms):
            Mx = ds1['cliq_mom'][:,2+i,t0:,:].reshape(ncases*(ntimes)*nalts)
            momscales[i] = np.max(Mx)
        for i in range(0,nmoms):
            for j in range(0,nrates):
                M[:,i,j]=M[:,i,j]/momscales[i]
                
        for i in range(0,M.shape[2]):
            # takes the log of everything but the 3rd moment process rate (which is all zeros for coll-coal)
            M[:,:,i],dMdt[:,:,i],masklog = rescalesinglecat(M[:,:,i],dMdt[:,:,i],i)
            if i == 0:
                maskall = masklog
            else:
                maskall = np.logical_and(maskall,masklog)
    else:
        M,dMdt = twocatdata(ds2)
        
        M,dMdt,mask0 = masktwocat(M,dMdt)
        
        nmoms = 3
        ncats = 2
        t0 =1
        
        momscales_c = np.zeros(nmoms)
        momscales_r = np.zeros(nmoms)
        
        for i in range(0,nmoms):
            Mx_c = ds2['cliq_mom'][:,i,t0:,:].reshape(ncases*(ntimes)*nalts)
            Mx_r = ds2['rain_mom'][:,i,t0:,:].reshape(ncases*(ntimes)*nalts)
            momscales_c[i] = np.max(Mx_c)
            momscales_r[i] = np.max(Mx_r)

            M[:,i,0]=M[:,i,0]/momscales_c[i]
            M[:,i,1]=M[:,i,1]/momscales_r[i]
    
        for j in range(0,M.shape[2]):
            M[:,:,j],masklog = rescaletwocat(M[:,:,j])
            if (j==0):
                maskall = masklog
            else:
                maskall = np.logical_and(maskall,masklog)
        
        
    ### Bin Data ###
    bin0,bin1coal,bin1condevap,binmag,momscales = bindatanormed(ds1,normalized=normalized)
    
    binmask = np.zeros(bin0.shape[0])

 
    #this gets rid of some random high values in the bin DSD's that throw off the training
    for i in range(0,bin0.shape[0]):
        if np.any(bin0[i,:,:]>1):
            binmask[i] = False
        else:
            binmask[i] = True
    
    maskall = np.logical_and(maskall,binmask)
            
    M = M[maskall,:]
    dMdt = dMdt[maskall,:]
    bin0 = bin0[maskall,:]
    bin1coal = bin1coal[maskall,:]
    bin1condevap = bin1condevap[maskall,:]
    binmag = binmag[maskall,:]
    
    binmagscales = np.zeros((2,6))

    
    
    for i in range(0,binmag.shape[1]):
        binmag[:,i] = np.log(binmag[:,i])
        #print(np.min(binmag[:,i]),np.max(binmag[:,i]))
        binmagscales[0,i]=np.min(binmag[:,i])
        binmagscales[1,i]=np.max(binmag[:,i])
        binmag[:,i] = (binmag[:,i]-np.min(binmag[:,i]))/(np.max(binmag[:,i])-np.min(binmag[:,i]))
        

    # this is because the vectors are modified in place in the rescaling function (probably fix this)
    onecatmoms,onecatrats = singlecatdata(ds1)
    # Load the two cat data
    twocatmoms,twocatrats=twocatdata(ds2)
    M1 = onecatmoms[:,:]
    dM1dt = onecatrats[:,:]
    
    M2 = twocatmoms[:,:]
    dM2dt = twocatrats[:,:]
    
    M1 = M1[maskall,:]
    dM1dt = dM1dt[maskall,:]
    M2 = M2[maskall,:]
    dM2dt = dM2dt[maskall,:]
    
    #return M,dMdt,bin0,bin1coal,bin1condevap,momscales,binmag,M1,dM1dt,M2,dM2dt
    return M,dMdt,bin0,bin1coal,bin1condevap,binmagscales,binmag,M1,dM1dt,M2,dM2dt
def scalemoments(moments):
    nmoms = 4
    nrates = 3

    minlogscale = np.zeros(nmoms)

    for i in range(0,nmoms):
        minlogscale[i] = np.min(moments[:,i,0])

    for i in range(0,nmoms):
        for j in range(0,nrates):
            moments[:,i,j]=moments[:,i,j]/(0.0-minlogscale[i])+1.0

    return moments,minlogscale

def unscalemoments(moments,minlogscale):
    nmoms = 4
    nrates = 3

    for i in range(0,nmoms):
        for j in range(0,nrates):
            moments[:,i,j]=(moments[:,i,j]-1.0)*(0.0-minlogscale[i])

    return moments
def diagnose_moments(x,m0,m3):
    ratio = m3/m0

    return m0*ratio**(x/3.0)
def getmoments_from_dist_np(distlm,distl,momscales):
    # from mp_bin.f90/moments_from_bins
    # distlm - mass bin distribution
    # distl - number bin distribution

    moms = np.arange(0,4)*3
    mc = np.zeros(moms.shape[0])

    ncd = 35
    rhow = 1000.0
    qfactor = 1.0 #np.pi/6*rhow# 1.0 # since the mass bins are saved as M3 not q
    m3 = distlm/qfactor

    # distl and distm are scaled by the maximum M0 and M3, so unscale them to calc. moments
    m0=distl*momscales[0]
    m3 = m3*momscales[1]


    for j in range(0,moms.shape[0]):
        mom = moms[j]

        mcc = 0.0
        for i in range(0,ncd):
            if distl[i]>0:
                mx = diagnose_moments(mom,m0[i],m3[i])

            else:
                mx = 0.0
            mcc = mcc+mx

        mc[j] = mcc

    # now rescale by the maximum moments
    mc=mc/momscales
    return mc
def bindatanormedcase(ds,case,normalized=True):
    # this returns the number and mass size distributions at this time step (t0) and the next time step (t1)
    # If normalized == True, normalized to add up to one
    # and the magnitude relative to the max in the data set is returned as a value between 0 and 1
    # case - the case to be selected

    ncases, ntimes, nalts, nbins = ds['dsd_number'].shape
    #print(ncases, ntimes, nalts, nbins)
    #dsd_number(case, time, altitude, bin_mass)
    t0 = 1
    ntimes-=1
    ncases = 1

    #this factor converts massbins to M3
    rhow = 1000.0
    factor = np.pi/6*rhow

    nbin0 = ds['dsd_number'][case,t0:-1,:,:].reshape(ncases*(ntimes-1)*nalts,nbins)
    mbin0 = ds['dsd_mass'][case,t0:-1,:,:].reshape(ncases*(ntimes-1)*nalts,nbins)/factor

    #dsd_mass(case, time, altitude, bin_mass)
    nbin1 = ds['dsd_number'][case,(t0+1):,:,:].reshape(ncases*(ntimes-1)*nalts,nbins)
    mbin1 = ds['dsd_mass'][case,(t0+1):,:,:].reshape(ncases*(ntimes-1)*nalts,nbins)/factor

    nbin1coal = nbin0+ds['dsd_number_coal'][case,t0:-1,:,:].reshape(ncases*(ntimes-1)*nalts,nbins)
    nbin1condevap = nbin0+ds['dsd_number_condevap'][case,t0:-1,:,:].reshape(ncases*(ntimes-1)*nalts,nbins)

    mbin1coal = mbin0+ds['dsd_mass_coal'][case,t0:-1,:,:].reshape(ncases*(ntimes-1)*nalts,nbins)/factor
    mbin1condevap = mbin0+ds['dsd_mass_condevap'][case,t0:-1,:,:].reshape(ncases*(ntimes-1)*nalts,nbins)/factor
            
    bin0 = np.zeros((ncases*(ntimes-1)*nalts,2,nbins))
    bin1coal = np.zeros((ncases*(ntimes-1)*nalts,2,nbins))
    bin1condevap = np.zeros((ncases*(ntimes-1)*nalts,2,nbins))
    binmag = np.zeros((ncases*(ntimes-1)*nalts,6))

    
    nbinsum = np.sum(nbin0,axis=1)
    mbinsum = np.sum(mbin0,axis=1)
    
    nbin1coalsum = np.sum(nbin1coal,axis=1)
    mbin1coalsum = np.sum(mbin1coal,axis=1)
    
    nbin1condevapsum = np.sum(nbin1condevap,axis=1)
    mbin1condevapsum = np.sum(mbin1condevap,axis=1)

    print(nbin0.shape)
    print(nbinsum.shape)
    if normalized:
        print("normalized")
        for k in range(0,nbins):
            #nbin0[:,k]=nbin1[:,k]/nbinsum[:]
            #mbin0[:,k]=mbin1coal[:,k]/mbinsum[:]
            nbin0[:,k]=nbin0[:,k]/nbinsum[:]
            mbin0[:,k]=mbin0[:,k]/mbinsum[:]

            nbin1coal[:,k]=nbin1coal[:,k]/nbinsum[:] #nbin1coalsum[:]
            mbin1coal[:,k]=mbin1coal[:,k]/mbinsum[:] #mbin1coalsum[:]

            nbin1condevap[:,k]=nbin1condevap[:,k]/nbin1condevapsum[:]
            mbin1condevap[:,k]=mbin1condevap[:,k]/mbin1condevapsum[:]
    
    
    bin0[:,0,:]=nbin0
    bin0[:,1,:]=mbin0

    bin1coal[:,0,:]=nbin1coal
    bin1coal[:,1,:]=mbin1coal

    bin1condevap[:,0,:]=nbin1condevap
    bin1condevap[:,1,:]=mbin1condevap

    binmag[:,0]=nbinsum
    binmag[:,1]=mbinsum
    binmag[:,2]=nbin1coalsum
    binmag[:,3]=mbin1coalsum
    binmag[:,4]=nbin1condevapsum
    binmag[:,5]=mbin1condevapsum
    
    nmoms = 4
    momscales = np.zeros(nmoms)

    for i in range(0,nmoms):
        Mx = ds['cliq_mom'][case,2+i,t0:,:].reshape(ncases*(ntimes)*nalts)
        momscales[i] = np.max(Mx)
    
    return bin0,bin1coal,bin1condevap,binmag,momscales

def flaggeddatanormedcase(ds1,ds2,case,binmagscales,normalized=True):
    # get rid of bad data and return the normalized data sets
    # case
    
    ### Single category data ###
    # Load the single cat data (Sample, moment, rate)
    M,dMdt = singlecatdatacase(ds1,case)
    # Flag data below thresholds and with no clouds
    
    M,dMdt,dMdtsign,mask0 = masksinglecat(M,dMdt)
    
    ### Bin Data ###
    bin0,bin1coal,bin1condevap,binmag,momscales = bindatanormedcase(ds1,case,normalized=normalized)
    
    #print(momscales)
    nmoms = 4
    nrates = 3
    
    binmask = np.zeros(bin0.shape[0])
    for i in range(0,nmoms):
        for j in range(0,nrates):
            M[:,i,j]=M[:,i,j]/momscales[i]
 
    #this gets rid of some random high values in the bin DSD's that throw off the training
    for i in range(0,bin0.shape[0]):
        if np.any(bin0[i,:,:]>1):
            binmask[i] = False
        else:
            binmask[i] = True

    for i in range(0,M.shape[2]):
        # takes the log of everything but the 3rd moment process rate (which is all zeros for coll-coal)
        M[:,:,i],dMdt[:,:,i],masklog = rescalesinglecat(M[:,:,i],dMdt[:,:,i],i)
        if i == 0:
            maskall = masklog
        else:
            maskall = np.logical_and(maskall,masklog)
    
    maskall = np.logical_and(maskall,binmask)
    
    for i in range(0,binmag.shape[1]):
        binmag[:,i] = np.log(binmag[:,i])
        #print(np.min(binmag[:,i]),np.max(binmag[:,i]))
        binmag[:,i] = (binmag[:,i]-binmagscales[0,i])/(binmagscales[1,i]-binmagscales[0,i])
        

    # this is because the vectors are modified in place in the rescaling function (probably fix this)
    onecatmoms,onecatrats = singlecatdatacase(ds1,case)
    # Load the two cat data
    twocatmoms,twocatrats=twocatdatacase(ds2,case)
    M1 = onecatmoms[:,:]
    dM1dt = onecatrats[:,:]
    
    M2 = twocatmoms[:,:]
    dM2dt = twocatrats[:,:]
    
    
    return M,dMdt,bin0,bin1coal,bin1condevap,momscales,binmag,M1,dM1dt,M2,dM2dt,maskall
def bindatacase(ds,case,momscales,rescaled=True):
    # this returns the number and mass size distributions at this time step (t0) and the next time step (t1)
    # for only a specific case - and also returns the time and height
    # If rescaled == True they are normalized by the sum over the bins at t0
    # and the magnitude relative to the max in the data set is returned as a value between 0 and 1

    ncases, ntimes, nalts, nbins = ds['dsd_number'].shape
    print(ncases, ntimes, nalts, nbins)
    #dsd_number(case, time, altitude, bin_mass)
    t0 = 1
    ntimes-=1
    ncases = 1

    #this factor converts massbins to M3
    rhow = 1000.0
    factor = np.pi/6*rhow

    nbin0 = ds['dsd_number'][case,t0:-1,:,:].reshape(ncases*(ntimes-1)*nalts,nbins)
    mbin0 = ds['dsd_mass'][case,t0:-1,:,:].reshape(ncases*(ntimes-1)*nalts,nbins)/factor

    #dsd_mass(case, time, altitude, bin_mass)
    nbin1 = ds['dsd_number'][case,(t0+1):,:,:].reshape(ncases*(ntimes-1)*nalts,nbins)
    mbin1 = ds['dsd_mass'][case,(t0+1):,:,:].reshape(ncases*(ntimes-1)*nalts,nbins)/factor

    nbin1coal = nbin0+ds['dsd_number_coal'][case,t0:-1,:,:].reshape(ncases*(ntimes-1)*nalts,nbins)
    nbin1condevap = nbin0+ds['dsd_number_condevap'][case,t0:-1,:,:].reshape(ncases*(ntimes-1)*nalts,nbins)

    mbin1coal = mbin0+ds['dsd_mass_coal'][case,t0:-1,:,:].reshape(ncases*(ntimes-1)*nalts,nbins)/factor
    mbin1condevap = mbin0+ds['dsd_mass_condevap'][case,t0:-1,:,:].reshape(ncases*(ntimes-1)*nalts,nbins)/factor

    nmoms = 4
    #momscales = np.zeros(nmoms)

    for i in range(0,nmoms):
        Mx = ds['cliq_mom'][case,2+i,t0:,:].reshape(ncases*(ntimes)*nalts)
        #momscales[i] = np.max(Mx)

    print(Mx.shape,momscales)

    if rescaled:
        # These lines scale the bin values by the sum
        for k in range(0,nbins):
            nbin0[:,k]=nbin0[:,k]/momscales[0]
            mbin0[:,k]=mbin0[:,k]/momscales[1]

            nbin1coal[:,k]=nbin1coal[:,k]/momscales[0]
            mbin1coal[:,k]=mbin1coal[:,k]/momscales[1]

            nbin1condevap[:,k]=nbin1condevap[:,k]/momscales[0]
            mbin1condevap[:,k]=mbin1condevap[:,k]/momscales[1]

    bin0 = np.zeros((ncases*(ntimes-1)*nalts,2,nbins))
    bin1coal = np.zeros((ncases*(ntimes-1)*nalts,2,nbins))
    bin1condevap = np.zeros((ncases*(ntimes-1)*nalts,2,nbins))
    binmag = np.zeros((ncases*(ntimes-1)*nalts,2))

    bin0[:,0,:]=nbin0
    bin0[:,1,:]=mbin0

    bin1coal[:,0,:]=nbin1coal
    bin1coal[:,1,:]=mbin1coal

    bin1condevap[:,0,:]=nbin1condevap
    bin1condevap[:,1,:]=mbin1condevap

    nbinsum = np.sum(nbin0,axis=1)
    mbinsum = np.sum(mbin0,axis=1)

    binmag[:,0]=nbinsum
    binmag[:,1]=mbinsum

    return bin0,bin1coal,bin1condevap,binmag,momscales
def singlecatdatacase(ds,case):
    # this returns the moments and rates for collision-coalescence for the single category case for t0

    proc_rates = ['cevap','ccoal','csed']
    nrates = len(proc_rates)
    momidxs = [2,3,4,5]
    ncases, nmoms, ntimes, nalts = ds['ccoal'].shape
    nmoms = len(momidxs)

    t0 = 1
    ntimes-=1
    ncases = 1

    singlecatrates = np.zeros((ncases*(ntimes-1)*nalts,nmoms,nrates))
    singlecatmoments = np.zeros((ncases*(ntimes-1)*nalts,nmoms,nrates))

    singlecatmoment = np.moveaxis(ds['cliq_mom'],1,3)
    singlecatmoment = singlecatmoment[case,:,:,:]
    singlecatmoment = singlecatmoment[t0:-1,:,momidxs]
    singlecatmoment = singlecatmoment.reshape(ncases*(ntimes-1)*nalts,nmoms)

    for i in range(0,nrates):
        proc_rate = proc_rates[i]
        singlecatrate = np.moveaxis(ds[proc_rate],1,3)
        singlecatrate = singlecatrate[case,:,:,:]
        singlecatrate = singlecatrate[t0:-1,:,momidxs]

        singlecatrate = singlecatrate.reshape(ncases*(ntimes-1)*nalts,nmoms)

        singlecatrates[:,:,i]=singlecatrate
        singlecatmoments[:,:,i]=singlecatmoment

        # correct the moments for the order the process rates are calculated in Tau
        singlecatmoment = singlecatmoment-singlecatrate

    return singlecatmoments,singlecatrates

def twocatdatacase(ds,case):
    cloud_rates = ['cauto','caccr','ccoal']
    rain_rates = ['auto','accr','coal']
    ncases,nmoms,ntimes,nalts = ds['ccoal'].shape
    nrates = len(cloud_rates)

    t0 = 1
    ntimes-=1
    ncases = 1

    #cliq_mom(case, cloud_moment, time, altitude)
    #rain_mom(case, rain_moment, time, altitude)
    #cauto(case, cloud_moment, time, altitude)
    #caccr(case, cloud_moment, time, altitude)
    #ccoal(case, cloud_moment, time, altitude)
    #auto(case, rain_moment, time, altitude)
    #accr(case, rain_moment, time, altitude)
    #coal(case, rain_moment, time, altitude)

    twocatrates = np.zeros((ncases*(ntimes-1)*nalts,nmoms,nrates,2))
    twocatmoments = np.zeros((ncases*(ntimes-1)*nalts,nmoms,nrates,2))

    cloudmoments = np.moveaxis(ds['cliq_mom'],1,3)
    rainmoments = np.moveaxis(ds['rain_mom'],1,3)
    cloudmoments = cloudmoments[case,:,:,:]
    cloudmoments = cloudmoments[t0:-1,:,:].reshape(ncases*(ntimes-1)*nalts,nmoms)
    rainmoments = rainmoments[case,:,:,:]
    rainmoments = rainmoments[t0:-1,:,:].reshape(ncases*(ntimes-1)*nalts,nmoms)

    for i in range(0,nrates):
        cloud_rate = cloud_rates[i]
        rain_rate = rain_rates[i]
        #print(cloud_rate,rain_rate)
        cloudrate = np.moveaxis(ds[cloud_rate],1,3)
        cloudrate = cloudrate[case,:,:,:]
        cloudrate = cloudrate[t0:-1,:,:].reshape(ncases*(ntimes-1)*nalts,nmoms)
        
        rainrate = np.moveaxis(ds[rain_rate],1,3)
        rainrate = rainrate[case,:,:,:]
        rainrate = rainrate[t0:-1,:,:].reshape(ncases*(ntimes-1)*nalts,nmoms)

        for j in range(0,nmoms):
            twocatrates[:,j,i,0]=cloudrate[:,j]
            twocatrates[:,j,i,1]=rainrate[:,j]
            twocatmoments[:,j,i,0]=cloudmoments[:,j]
            twocatmoments[:,j,i,1]=rainmoments[:,j]

    return twocatmoments,twocatrates