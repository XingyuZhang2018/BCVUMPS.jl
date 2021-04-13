using Random, Statistics, ProgressMeter

#2x2-J1-J2-checkboard
function e_dif(model::Ising22, array, lattice, x, y)
    r = model.r
    a , b = mod(x,2), mod(y,2)
    
    top     = array[x, y + 1 - lattice * (y==lattice)] 
    bottom  = array[x,y - 1 + lattice * (y==1)]
    left    = array[x - 1 + lattice * (x==1), y]
    right   = array[x + 1 - lattice * (x==lattice), y]
    
    if a==1
        right *= r
    else 
        left *= r
    end
    if b==1
        top *= r
    else 
        bottom *= r
    end
    
    return 2 * array[x, y] * (top+bottom+left+right)
end

#3x3-J1-J2
function e_dif(model::Ising33, array, lattice, x, y)
    r = model.r
    a , b = mod(x,3), mod(y,3)
    
    top     = array[x, y + 1 - lattice * (y==lattice)] 
    bottom  = array[x,y - 1 + lattice * (y==1)] 
    left    = array[x - 1 + lattice * (x==1), y]
    right   = array[x + 1 - lattice * (x==lattice), y]

    if a == 1
        if b == 2
            right *= r
        end
    elseif a == 2
        if b == 1
            top *= r
        elseif b == 2
            top *= r 
            bottom *= r
            left *= r
            right *= r
        else
            bottom *= r
        end
    else
        if b == 2
            left *= r 
        end
    end
        
    return 2 * array[x, y] * (top+bottom+left+right)
end

function onestep!(model::MT, spin_array, lattice, β) where {MT <: HamiltonianModel}
    for i = 1:lattice
        for j = 1:lattice
            e = e_dif(model, spin_array, lattice, i, j)
            if e <= 0
                spin_array[i,j] = -spin_array[i,j];
            elseif exp(-e*β) > rand()
                spin_array[i,j] = -spin_array[i,j];
            end
        end
    end
    return spin_array
end

function energy(model::MT, spin_array, lattice) where {MT <: HamiltonianModel}
    tol_en = 0
    for i = 1:lattice
        for j = 1:lattice
            e = e_dif(model, spin_array, lattice, i, j)
            tol_en += e
        end
    end
    return -tol_en/4
end

function MCMC(model::MT,lattice,β,Sweeps_heat,Sweeps) where {MT <: HamiltonianModel}
#     spin_array = ones(lattice,lattice)  
    spin_array = (bitrand(lattice,lattice).-0.5)*2 
    for j = 1:Sweeps_heat
        spin_array = onestep!(model,spin_array,lattice ,β);
    end
    mag = zeros(1,Sweeps);
    ene = zeros(1,Sweeps)
    Threads.@threads for j = 1:Sweeps
        spin_array = onestep!(model,spin_array,lattice ,β)
        mag[j] = abs(sum(spin_array)/lattice^2)
        ene[j] = energy(model,spin_array, lattice)/lattice^2
    end
    mag_ave = sum(mag)/Sweeps;
    ene_ave = sum(ene)/Sweeps;
    return mag_ave, ene_ave
end 

function mutiMC(model::MT,β,lattice,bins,Sweeps_heat,Sweeps) where {MT <: HamiltonianModel}
    eMC_bins = zeros(bins,1)
    Threads.@threads for j=1:bins
        eMC_bins[j] =  MCMC(model,lattice,β,Sweeps_heat,Sweeps)
    end
    eMC = mean(eMC_bins)
    return eMC
end

function MutiMC(model::MT,β,lattice,Bins,bins,Sweeps_heat,Sweeps) where {MT <: HamiltonianModel}
    eMC_Bins = zeros(1,Bins)
    p = Progress(Bins)
    for j = 1:Bins
        eMC_Bins[j] = mutiMC(model,β,lattice,bins,Sweeps_heat,Sweeps)
        update!(p,j)
    end
    eMC_m = mean(eMC_Bins)
    err = 1.96*sqrt(var(eMC_Bins)/Bins)
    return eMC_m,err
end