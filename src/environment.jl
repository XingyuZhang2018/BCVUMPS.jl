using LinearAlgebra
using KrylovKit
using Random

"""
    i, j = ktoij(k,Ni,Nj)

    LinearIndices -> CartesianIndices
"""
function ktoij(k,Ni,Nj)
    Cart = CartesianIndices((1:Ni,1:Nj))
    Index = Cart[k]
    Index[1],Index[2]
end

safesign(x::Number) = iszero(x) ? one(x) : sign(x)
"""
    qrpos(A)

Returns a QR decomposition, i.e. an isometric `Q` and upper triangular `R` matrix, where `R`
is guaranteed to have positive diagonal elements.
"""
qrpos(A) = qrpos!(copy(A))
function qrpos!(A)
    F = qr!(A)
    Q = Matrix(F.Q)
    R = F.R
    phases = safesign.(diag(R))
    rmul!(Q, Diagonal(phases))
    lmul!(Diagonal(conj!(phases)), R)
    return Q, R
end

"""
    lqpos(A)

Returns a LQ decomposition, i.e. a lower triangular `L` and isometric `Q` matrix, where `L`
is guaranteed to have positive diagonal elements.
"""
lqpos(A) = lqpos!(copy(A))
function lqpos!(A)
    F = qr!(Matrix(A'))
    Q = Matrix(Matrix(F.Q)')
    L = Matrix(F.R')
    phases = safesign.(diag(L))
    lmul!(Diagonal(phases), Q)
    rmul!(L, Diagonal(conj!(phases)))
    return L, Q
end

function cellones(Ni,Nj,D)
    Cell = Array{Array{Float64,2},2}(undef, Ni, Nj)
    Threads.@threads for k = 1:Ni*Nj
        i,j = ktoij(k, Ni, Nj)
        Cell[i,j] = Matrix{Float64}(I, D, D)
    end
    return Cell
end

function ρmap(ρ,Ai,J)
    Nj = size(Ai,1)
    for j = 1:Nj
        jr = J+j-1 - (J+j-1 > Nj)*Nj
        ρ = ein"dc,csb,dsa -> ab"(ρ,Ai[jr],conj(Ai[jr]))
        # @tensor ρ[a,b] := ρ[a',b']*Ai[jr][b',s,b]*conj(Ai[jr][a',s,a])
    end
    return ρ
end

function initialA(M, D)
    Ni, Nj = size(M)
    T = eltype(M[1,1])
    A = Array{Array{Float64,3},2}(undef, Ni, Nj)
    Threads.@threads for k = 1:Ni*Nj
        i,j = ktoij(k, Ni, Nj)
        d = size(M[i,j], 4)
        A[i,j] = rand(T, D, d, D)
    end
    return A
end
"""
    getL!(A,L; kwargs...)

````
┌ A1─A2─    ┌      L ─
ρ │  │    = ρ   =  │
┕ A1─A2─    ┕      L'─
````

ρ=L'*L, return L, where `L`is guaranteed to have positive diagonal elements.
L = cholesky!(ρ).U
If ρ is not exactly positive definite, cholesky will fail
"""
function getL!(A,L; kwargs...)
    Ni,Nj = size(A)
    # for j = 1:Nj, i = 1:Ni
    Threads.@threads for k = 1:Ni*Nj
        i,j = ktoij(k, Ni, Nj)
        _,ρs,_ = eigsolve(ρ->ρmap(ρ,A[i,:],j), L[i,j]'*L[i,j], 1, :LM; ishermitian = false, maxiter = 1, kwargs...)
        ρ = real(ρs[1] + ρs[1]')
        ρ ./= tr(ρ)
        F = svd!(ρ)
        Lo = lmul!(Diagonal(sqrt.(F.S)), F.Vt)
        _, L[i,j] = qrpos!(Lo)
    end
    return L
end

"""
    getAL(A,L)

Given an MPS tensor `A` and `L` ，return a left-canonical MPS tensor `AL`, a gauge transform `R` and
a scalar factor `λ` such that ``λ AR R = L A``
"""
function getAL(A,L)
    Ni,Nj = size(A)
    AL = Array{Array{Float64,3},2}(undef, Ni, Nj)
    Le = Array{Array{Float64,2},2}(undef, Ni, Nj)
    λ = zeros(Ni,Nj)
    Threads.@threads for k = 1:Ni*Nj
        i,j = ktoij(k, Ni, Nj)
        D, d, = size(A[i,j])
        Q, R = qrpos!(reshape(L[i,j]*reshape(A[i,j], D, d*D), D*d, D))
        AL[i,j] = reshape(Q, D, d, D)
        λ[i,j] = norm(R)
        Le[i,j] = rmul!(R, 1/λ[i,j])
    end
    return AL, Le, λ
end

function getLsped(Le, A, AL; kwargs...)
    Ni,Nj = size(A)
    L = Array{Array{Float64,2},2}(undef, Ni, Nj)
    Threads.@threads for k = 1:Ni*Nj
        i,j = ktoij(k, Ni, Nj)
        _, Ls, _ = eigsolve(X -> ein"dc,csb,dsa -> ab"(X,A[i,j],conj(AL[i,j])), Le[i,j], 1, :LM; ishermitian = false, kwargs...)
        _, L[i,j] = qrpos!(real(Ls[1]))
    end
    return L
end

"""
    leftorth(A,L=cellones(size(A,1),size(A,2),size(A[1,1],1)); tol = 1e-12, maxiter = 100, kwargs...)

Given an MPS tensor `A`, return a left-canonical MPS tensor `AL`, a gauge transform `L` and
a scalar factor `λ` such that ``λ AL L = L A``, where an initial guess for `L` can be
provided.
"""
function leftorth(A,L=cellones(size(A,1),size(A,2),size(A[1,1],1)); tol = 1e-12, maxiter = 100, kwargs...)
    L = getL!(A,L; kwargs...)
    AL, Le, λ= getAL(A,L;kwargs...)
    numiter = 1
    while norm(L.-Le) > tol && numiter < maxiter
        L = getLsped(Le, A, AL; kwargs...)
        AL, Le, λ = getAL(A, L; kwargs...)
        numiter += 1
    end
    L = Le
    return AL, L, λ
end


"""
    rightorth(A,L=cellones(size(A,1),size(A,2),size(A[1,1],1)); tol = 1e-12, maxiter = 100, kwargs...)

Given an MPS tensor `A`, return a gauge transform R, a right-canonical MPS tensor `AR`, and
a scalar factor `λ` such that ``λ R AR^s = A^s R``, where an initial guess for `R` can be
provided.
"""
function rightorth(A,L=cellones(size(A,1),size(A,2),size(A[1,1],1)); tol = 1e-12, maxiter = 100, kwargs...)
    Ni,Nj = size(A)
    Ar = Array{Array{Float64,3},2}(undef, Ni, Nj)
    Lr = Array{Array{Float64,2},2}(undef, Ni, Nj)
    Threads.@threads for k = 1:Ni*Nj
        i,j = ktoij(k, Ni, Nj)
        Ar[i,j] = permutedims(A[i,j],(3,2,1))
        Lr[i,j] = permutedims(L[i,j],(2,1))
    end
    AL, L, λ = leftorth(Ar,Lr; tol = tol, kwargs...)
    R = Array{Array{Float64,2},2}(undef, Ni, Nj)
    AR = Array{Array{Float64,3},2}(undef, Ni, Nj)
    Threads.@threads for k = 1:Ni*Nj
        i,j = ktoij(k, Ni, Nj)
        R[i,j] = permutedims(L[i,j],(2,1))
        AR[i,j] = permutedims(AL[i,j],(3,2,1))
    end
    return R, AR, λ
end

"""
    LRtoC(L,R)

```
 ── Cᵢⱼ ──  =  ── Lᵢⱼ ── Rᵢⱼ₊₁ ──
```
"""
function LRtoC(L,R)
    Ni, Nj = size(L)
    C = Array{Array{Float64,2},2}(undef, Ni, Nj)
    Threads.@threads for k = 1:Ni*Nj
        i,j = ktoij(k, Ni, Nj)
        jr = j + 1 - (j + 1 > Nj) * Nj
        C[i,j] = L[i,j] * R[i,jr]
    end
    return C
end

"""
    FLm = FLmap(ALi, ALip, Mi, FL, J)

ALip means ALᵢ₊₁
```
  ┌──        ┌──  ALᵢⱼ  ── ALᵢⱼ₊₁   ──   ...   
  │          │     │        │          
 FLm   =   FLᵢⱼ ─ Mᵢⱼ   ── Mᵢⱼ₊₁     ──   ...  
  │          │     │        │       
  ┕──        ┕──  ALᵢ₊₁ⱼ ─ ALᵢ₊₁ⱼ₊₁ ──   ...
```
"""
function FLmap(ALi, ALip, Mi, FL, J)
    Nj = size(ALi,1)
    FLm = copy(FL)
    for j=1:Nj
        jr = J+j-1 - (J+j-1 > Nj)*Nj
        FLm = ein"abc,cde,bfhd,afg -> ghe"(FLm,ALi[jr],Mi[jr],conj(ALip[jr]))
    end
    return FLm
end

"""
    FRm = FRmap(ARi, ARip, Mi, FR, J)

ARip means ARᵢ₊₁
```
 ──┐       ... ─── ARᵢⱼ₋₁  ── ARᵢⱼ  ──┐ 
   │                │          │      │ 
──FRm  =   ... ──── Mᵢⱼ₋₁  ── Mᵢⱼ  ──FRᵢⱼ
   │                │          │      │  
 ──┘       ... ─ ARᵢ₊₁ⱼ₋₁ ─ ARᵢ₊₁ⱼ  ──┘ 
```
"""
function FRmap(ARi, ARip, Mi, FR, J)
    Nj = size(ARi,1)
    FRm = copy(FR)
    for j=1:Nj
        jr = J-(j-1) + (J-(j-1) < 1)*Nj
        FRm = ein"abc,eda,hfbd,gfc -> ehg"(FRm,ARi[jr],Mi[jr],conj(ARip[jr]))
    end
    return FRm
end

function FLint(AL, M)
    Ni,Nj = size(AL)
    FL = Array{Array{Float64,3},2}(undef, Ni, Nj)
    Threads.@threads for k = 1:Ni*Nj
        i,j = ktoij(k, Ni, Nj)
        D = size(AL[i,j],1)
        dL = size(M[i,j],1)
        FL[i,j] = rand(Float64, D, dL, D)
    end
    return FL
end

function FRint(AR, M)
    Ni,Nj = size(AR)
    FR = Array{Array{Float64,3},2}(undef, Ni, Nj)
    Threads.@threads for k = 1:Ni*Nj
        i,j = ktoij(k, Ni, Nj)
        D = size(AR[i,j],1)
        dR = size(M[i,j],3)
        FR[i,j] = rand(Float64, D, dR, D)
    end
    return FR
end

"""
    λL, FL = leftenv(AL, M, FL = FLint(AL,M); kwargs...) 

Compute the left environment tensor for MPS A and MPO M, by finding the left fixed point
of AL - M - conj(AL) contracted along the physical dimension.
```
 ┌──  ALᵢⱼ  ── ALᵢⱼ₊₁   ──   ...         ┌── 
 │     │        │                        │   
FLᵢⱼ ─ Mᵢⱼ  ── Mᵢⱼ₊₁    ──   ...  = λLᵢⱼ FLᵢⱼ 
 │     │        │                        │   
 ┕──  ALᵢ₊₁ⱼ ─ ALᵢ₊₁ⱼ₊₁ ──   ...         ┕── 
```
"""
leftenv(AL, M, FL = FLint(AL,M); kwargs...) = leftenv!(AL, M, copy(FL); kwargs...)
function leftenv!(AL, M, FL; kwargs...)
    Ni,Nj = size(AL)
    λL = zeros(Ni,Nj)
    Threads.@threads for k = 1:Ni*Nj
        i,j = ktoij(k, Ni, Nj)
        ir = i + 1 - Ni * (i==Ni)
        λLs, FL1s, _= eigsolve(X->FLmap(AL[i,:], AL[ir,:], M[i,:], X, j), FL[i,j], 1, :LM; ishermitian = false, kwargs...)
        if length(λLs) > 1 && norm(abs(λLs[1]) - abs(λLs[2])) < 1e-12
            @show λLs
            if real(λLs[1]) > 0
                FL[i,j] = real(FL1s[1])
                λL[i,j] = real(λLs[1])
            else
                FL[i,j] = real(FL1s[2])
                λL[i,j] = real(λLs[2])
            end
        else
            FL[i,j] = real(FL1s[1])
            λL[i,j] = real(λLs[1])
        end
    end
    return λL, FL
end

"""
    λR, FR = rightenv(AR, M, FR = FRint(AR,M); kwargs...)

Compute the right environment tensor for MPS A and MPO M, by finding the left fixed point
of AR - M - conj(AR) contracted along the physical dimension.
```
   ... ─── ARᵢⱼ₋₁  ── ARᵢⱼ  ──┐          ──┐   
            │          │      │            │  
   ... ──── Mᵢⱼ₋₁  ── Mᵢⱼ  ──FRᵢⱼ  = λRᵢⱼ──FRᵢⱼ
            │          │      │            │  
   ... ─ ARᵢ₊₁ⱼ₋₁ ─ ARᵢ₊₁ⱼ  ──┘          ──┘  
```
"""
rightenv(AR, M, FR = FRint(AR,M); kwargs...) = rightenv!(AR, M, copy(FR); kwargs...)
function rightenv!(AR, M, FR; kwargs...)
    Ni,Nj = size(AR)
    λR = zeros(Ni,Nj)
    Threads.@threads for k = 1:Ni*Nj
        i,j = ktoij(k, Ni, Nj)
        ir = i + 1 - Ni * (i==Ni)
        λRs, FR1s, _= eigsolve(X->FRmap(AR[i,:], AR[ir,:], M[i,:], X, j), FR[i,j], 1, :LM; ishermitian = false, kwargs...)
        if length(λRs) > 1 && norm(abs(λRs[1]) - abs(λRs[2])) < 1e-12
            @show λRs
            if real(λRs[1]) > 0
                FR[i,j] = real(FR1s[1])
                λR[i,j] = real(λRs[1])
            else
                FR[i,j] = real(FR1s[2])
                λR[i,j] = real(λRs[2])
            end
        else
            FR[i,j] = real(FR1s[1])
            λR[i,j] = real(λRs[1])
        end
    end
    return λR, FR
end

"""
    ACm = ACmap(ACij, FLj, FRj, Mj, II)

```
                                ┌─────── ACᵢⱼ ─────┐
                                │        │         │          
┌─────── ACm  ─────┐      =     FLᵢⱼ ─── Mᵢⱼ ───── FRᵢⱼ
│        │         │            │        │         │   
                                FLᵢ₊₁ⱼ ─ Mᵢ₊₁ⱼ ──  FRᵢ₊₁ⱼ
                                │        │         │    
                                .        .         .
                                .        .         .
                                .        .         .
```
"""
function ACmap(ACij, FLj, FRj, Mj, II)
    Ni = size(FLj,1)
    ACm = copy(ACij)
    for i=1:Ni
        ir = II+i-1 - (II+i-1 > Ni)*Ni
        ACm = ein"abc,cde,bhfd,efg -> ahg"(FLj[ir],ACm,Mj[ir],FRj[ir])
    end
    return ACm
end

"""
    Cmap(Cij, FLjp, FRj, II)

```
                    ┌────Cᵢⱼ ───┐
                    │           │          
┌──── Cm ───┐   =   FLᵢⱼ₊₁ ──── FRᵢⱼ
│           │       │           │   
                    FLᵢ₊₁ⱼ₊₁ ── FRᵢ₊₁ⱼ
                    │           │        
                    .           .     
                    .           .     
                    .           .     
```
"""
function Cmap(Cij, FLjp, FRj, II)
    Ni = size(FLjp,1)
    Cm = copy(Cij)
    for i=1:Ni
        ir = II+i-1 - (II+i-1 > Ni)*Ni
        Cm = ein"abc,cd,dbe -> ae"(FLjp[ir],Cm,FRj[ir])
    end
    return Cm
end

"""
    ACenv(AC, FL, M, FR;kwargs...)

Compute the up environment tensor for MPS `FL`,`FR` and MPO `M`, by finding the up fixed point
        of `FL - M - FR` contracted along the physical dimension.
```
┌─────── ACᵢⱼ ─────┐
│        │         │          
FLᵢⱼ ─── Mᵢⱼ ───── FRᵢⱼ
│        │         │   
FLᵢ₊₁ⱼ ─ Mᵢ₊₁ⱼ ──  FRᵢ₊₁ⱼ  =  λACᵢⱼ ┌──── ACᵢⱼ ───┐
│        │         │                │      │      │  
.        .         .
.        .         .
.        .         .
```
"""
ACenv(AC, FL, M, FR; kwargs...) = ACenv!(copy(AC), FL, M, FR; kwargs...)
function ACenv!(AC, FL, M, FR; kwargs...)
    Ni,Nj = size(AC)
    λAC = zeros(Ni,Nj)
    Threads.@threads for k = 1:Ni*Nj
        i,j = ktoij(k, Ni, Nj)
        λACs, ACs, = eigsolve(X->ACmap(X, FL[:,j], FR[:,j], M[:,j], i), AC[i,j], 1, :LM; ishermitian = false, kwargs...)
        if length(λACs) > 1 && norm(abs(λACs[1]) - abs(λACs[2])) < 1e-12
            @show λACs
            if real(λACs[1]) > 0
                AC[i,j] = real(ACs[1])
                λAC[i,j] = real(λACs[1])
            else
                AC[i,j] = real(ACs[2])
                λAC[i,j] = real(λACs[2])
            end
        else
            AC[i,j] = real(ACs[1])
            λAC[i,j] = real(λACs[1])
        end
    end
    return λAC, AC
end

"""
    Cenv(C, FL, FR;kwargs...)

Compute the up environment tensor for MPS `FL` and `FR`, by finding the up fixed point
    of `FL - FR` contracted along the physical dimension.
```
┌────Cᵢⱼ ───┐
│           │          
FLᵢⱼ₊₁ ──── FRᵢⱼ
│           │   
FLᵢ₊₁ⱼ₊₁ ── FRᵢ₊₁ⱼ   =  λCᵢⱼ ┌──Cᵢⱼ ─┐
│           │                │       │  
.           .     
.           .     
.           .     
```
"""
Cenv(C, FL, FR; kwargs...) = Cenv!(copy(C), FL, FR; kwargs...)
function Cenv!(C, FL, FR; kwargs...)
    Ni,Nj = size(C)
    λC = zeros(Ni,Nj)
    Threads.@threads for k = 1:Ni*Nj
        i,j = ktoij(k, Ni, Nj)
        jr = j + 1 - (j==Nj) * Nj
        λCs, Cs, = eigsolve(X->Cmap(X, FL[:,jr], FR[:,j], i), C[i,j], 1, :LM; ishermitian = false, kwargs...)
        if length(λCs) > 1 && norm(abs(λCs[1]) - abs(λCs[2])) < 1e-12
            @show λCs
            if real(λCs[1]) > 0
                C[i,j] = real(Cs[1])
                λC[i,j] = real(λCs[1])
            else
                C[i,j] = real(Cs[2])
                λC[i,j] = real(λCs[2])
            end
        else
            C[i,j] = real(Cs[1])
            λC[i,j] = real(λCs[1])
        end
    end
    return λC, C
end

function ACCtoAL(ACij,Cij)
    D,d, = size(ACij)
    QAC, _ = qrpos(reshape(ACij,(D*d, D)))
    QC, _ = qrpos(Cij)
    reshape(QAC*QC', (D, d, D))
end

function ACCtoAR(ACij,Cijr)
    D,d, = size(ACij)
    _, QAC = lqpos(reshape(ACij,(D, d*D)))
    _, QC = lqpos(Cijr)
    reshape(QC'*QAC, (D, d, D))
end

"""
    itoir(i,Ni,Nj)

````
i -> (i,j) -> (i,jr) -> ir
````
"""
function itoir(i,Ni,Nj)
    Liner = LinearIndices((1:Ni,1:Nj))
    Cart = CartesianIndices((1:Ni,1:Nj))
    Index = Cart[i]
    i,j = Index[1],Index[2]
    jr = j - 1 + (j==1)*Nj
    Liner[i,jr]
end

function ALCtoAC(AL,C)
    Ni,Nj = size(AL)
    ACij = [ein"asc,cb -> asb"(AL[i],C[i]) for i=1:Ni*Nj]
    reshape(ACij,Ni,Nj)
end

"""
    ACCtoALAR(AL, C, AR, M, FL, FR; kwargs...)

QR factorization to get `AL` and `AR` from `AC` and `C`

````
──ALᵢⱼ──Cᵢⱼ──  =  ──ACᵢⱼ──  = ──Cᵢ₋₁ⱼ ──ARᵢⱼ──
  │                  │                  │   
````
"""
function ACCtoALAR(AL, C, AR, M, FL, FR; kwargs...)
    Ni,Nj = size(AL)
    AC = ALCtoAC(AL,C)
    _, AC = ACenv(AC, FL, M, FR; kwargs...)
    _, C = Cenv(C, FL, FR; kwargs...)

    ALij = [ACCtoAL(AC[i],C[i]) for i=1:Ni*Nj]
    AL = reshape(ALij,Ni,Nj)
    ARij = [ACCtoAR(AC[i],C[itoir(i,Ni,Nj)]) for i=1:Ni*Nj]
    AR = reshape(ARij,Ni,Nj)
    return AL, C, AR
end

"""
Compute the error through all environment `AL,C,FL,M,FR`

````
        ┌── AC──┐         
        │   │   │           ┌── AC──┐ 
MAC1 =  FL─ M ──FR  =  λAC  │   │   │ 
        │   │   │         

        ┌── AC──┐         
        │   │   │           ┌──C──┐ 
MAC2 =  FL─ M ──FR  =  λAC  │     │ 
        │   │   │         
        ┕── AL─     
        
── MAC1 ──    ≈    ── AL ── MAC2 ── 
    │                 │
````
"""
function error(AL,C,FL,M,FR)
    Ni,Nj = size(AL)
    AC = Array{Array{Float64,3},2}(undef, Ni,Nj)
    err = 0
    Threads.@threads for k = 1:Ni*Nj
        i,j = ktoij(k, Ni, Nj)
        AC[i,j] = ein"asc,cb -> asb"(AL[i,j],C[i,j])
        MAC = ACmap(AC[i,j], FL[:,j], FR[:,j], M[:,j], i)
        MAC -= ein"asd,cpd,cpb -> asb"(AL[i,j],conj(AL[i,j]),MAC)
        err += norm(MAC)
    end
    return err
end

"""
    obs_env()

If `Ni,Nj>1` and `Mij` are different bulk tensor, the up and down environment are different. So to calculate observable, we must get ACup and ACdown, which is easy to get by overturning the `Mij`. Then be cautious to get the new `FL` and `FR` environment.
"""
function obs_env() end

"""
    λL, FL = obs2x2FL(AL, M, FL = FLint(AL,M); kwargs...)

This function is designed specifically for 2x2 longitudinally symmetric cell to get correct `FL` environment.
```
 ┌──  ALᵢⱼ  ── ALᵢⱼ₊₁   ──   ...         ┌── 
 │     │        │                        │   
FLᵢⱼ ─ Mᵢⱼ  ── Mᵢⱼ₊₁    ──   ...  = λLᵢⱼ FLᵢⱼ 
 │     │        │                        │   
 ┕──  ALᵢⱼ  ─  ALᵢⱼ₊₁   ──   ...         ┕── 
```
"""
obs2x2FL(AL, M, FL = FLint(AL,M); kwargs...) = obs2x2FL!(AL, M, copy(FL); kwargs...)
function obs2x2FL!(AL, M, FL; kwargs...)
    Ni,Nj = size(AL)
    λL = zeros(Ni,Nj)
    Threads.@threads for k = 1:Ni*Nj
        i,j = ktoij(k, Ni, Nj)
        λLs, FL1s, _= eigsolve(X->FLmap(AL[i,:], AL[i,:], M[i,:], X, j), FL[i,j], 1, :LM; ishermitian = false, kwargs...)
        if length(λLs) > 1 && norm(abs(λLs[1]) - abs(λLs[2])) < 1e-12
            @show λLs
            if real(λLs[1]) > 0
                FL[i,j] = real(FL1s[1])
                λL[i,j] = real(λLs[1])
            else
                FL[i,j] = real(FL1s[2])
                λL[i,j] = real(λLs[2])
            end
        else
            FL[i,j] = real(FL1s[1])
            λL[i,j] = real(λLs[1])
        end
    end
    return λL, FL
end

"""
    λR, FR = obs2x2FR(AR, M, FR = FRint(AR,M); kwargs...)

This function is designed specifically for 2x2 longitudinally symmetric cell to get correct `FL` environment.
```
   ... ─── ARᵢⱼ₋₁  ── ARᵢⱼ  ──┐          ──┐   
            │          │      │            │  
   ... ──── Mᵢⱼ₋₁  ── Mᵢⱼ  ──FRᵢⱼ  = λRᵢⱼ──FRᵢⱼ
            │          │      │            │  
   ... ─── ARᵢⱼ₋₁  ── ARᵢⱼ  ──┘          ──┘  
```
"""
obs2x2FR(AR, M, FR = FRint(AR,M); kwargs...) = obs2x2FR!(AR, M, copy(FR); kwargs...)
function obs2x2FR!(AR, M, FR; kwargs...)
    Ni,Nj = size(AR)
    λR = zeros(Ni,Nj)
    Threads.@threads for k = 1:Ni*Nj
        i,j = ktoij(k, Ni, Nj)
        λRs, FR1s, _= eigsolve(X->FRmap(AR[i,:], AR[i,:], M[i,:], X, j), FR[i,j], 1, :LM; ishermitian = false, kwargs...)
        if length(λRs) > 1 && norm(abs(λRs[1]) - abs(λRs[2])) < 1e-12
            @show λRs
            if real(λRs[1]) > 0
                FR[i,j] = real(FR1s[1])
                λR[i,j] = real(λRs[1])
            else
                FR[i,j] = real(FR1s[2])
                λR[i,j] = real(λRs[2])
            end
        else
            FR[i,j] = real(FR1s[1])
            λR[i,j] = real(λRs[1])
        end
    end
    return λR, FR
end

function BgFLint(AL, M)
    Ni,Nj = size(AL)
    BgFL = Array{Array{Float64,4},2}(undef, Ni, Nj)
    Threads.@threads for k = 1:Ni*Nj
        i,j = ktoij(k, Ni, Nj)
        ir = i + 1 - Ni * (i==Ni)
        irr = i + 2 - Ni * (i + 2 > Ni)
        D1 = size(AL[i,j],1)
        D2 = size(AL[irr,j],1)
        dL1 = size(M[i,j],1)
        dL2 = size(M[ir,j],1)
        BgFL[i,j] = rand(Float64, D1, dL1, dL2, D2)
    end
    return BgFL
end

"""
    BgFLm = BgFLmap(ALi, ALip, Mi, Mip, BgFLij, J)

ALip means ALᵢ₊₂
Mip means Mᵢ₊₁
```
  ┌──        ┌──  ALᵢⱼ  ── ALᵢⱼ₊₁    ──   ...   
  │          │     │        │          
  │          │  ─ Mᵢⱼ   ── Mᵢⱼ₊₁     ──   ...  
BgFLm   =  BgFLᵢⱼ  │        │       
  │          │  ─ Mᵢ₊₁ⱼ ── Mᵢ₊₁ⱼ₊₁   ──   ...  
  │          │     │        │       
  ┕──        ┕──  ALᵢ₊₂ⱼ ─ ALᵢ₊₂ⱼ₊₁  ──   ... 
```
"""
function BgFLmap(ALi, ALip, Mi, Mip, BgFLij, J)
    Nj = size(ALi,1)
    BgFLm = copy(BgFLij)
    for j=1:Nj
        jr = J+j-1 - (J+j-1 > Nj)*Nj
        BgFLm = ein"dcba,def,ckge,bjhk,aji -> fghi"(BgFLm,ALi[jr],Mi[jr],Mip[jr],conj(ALip[jr]))
    end
    return BgFLm
end

"""
    λL, BgFL = bigleftenv(AL, M, BgFL = BgFLint(AL,M); kwargs...)  

Compute the left environment tensor for MPS A and MPO M, by finding the left fixed point
of AL - M - M - conj(AL) contracted along the physical dimension.
```
   ┌──  ALᵢⱼ  ── ALᵢⱼ₊₁    ──   ...           ┌── 
   │     │        │                           │   
   │  ─ Mᵢⱼ   ── Mᵢⱼ₊₁     ──   ...           │   
 BgFLᵢⱼ  │        │                   = λLᵢⱼ BgFLᵢⱼ
   │  ─ Mᵢ₊₁ⱼ ── Mᵢ₊₁ⱼ₊₁   ──   ...           │   
   │     │        │                           │   
   ┕──  ALᵢ₊₂ⱼ ─ ALᵢ₊₂ⱼ₊₁  ──   ...           ┕── 
```
"""
bigleftenv(AL, M, BgFL = BgFLint(AL,M); kwargs...) = bigleftenv!(AL, M, copy(BgFL); kwargs...)
function bigleftenv!(AL, M, BgFL; kwargs...)
    Ni,Nj = size(AL)
    λL = zeros(Ni,Nj)
    Threads.@threads for k = 1:Ni*Nj
        i,j = ktoij(k, Ni, Nj)
        ir = i + 1 - Ni * (i==Ni)
        # irr = i + 2 - Ni * (i + 2 > Ni) # modified for 2x2
        λLs, BgFL1s, _= eigsolve(X->BgFLmap(AL[i,:], AL[ir,:], M[i,:], M[ir,:], X, j), BgFL[i,j], 1, :LM; ishermitian = false, kwargs...)
        if length(λLs) > 1 && norm(abs(λLs[1]) - abs(λLs[2])) < 1e-12
            @show λLs
            if real(λLs[1]) > 0
                BgFL[i,j] = real(BgFL1s[1])
                λL[i,j] = real(λLs[1])
            else
                BgFL[i,j] = real(BgFL1s[2])
                λL[i,j] = real(λLs[2])
            end
        else
            BgFL[i,j] = real(BgFL1s[1])
            λL[i,j] = real(λLs[1])
        end
    end
    return λL, BgFL
end

function BgFRint(AR, M)
    Ni,Nj = size(AR)
    BgFR = Array{Array{Float64,4},2}(undef, Ni, Nj)
    Threads.@threads for k = 1:Ni*Nj
        i,j = ktoij(k, Ni, Nj)
        ir = i + 1 - Ni * (i==Ni)
        irr = i + 2 - Ni * (i + 2 > Ni)
        D1 = size(AR[i,j],3)
        D2 = size(AR[irr,j],3)
        dR1 = size(M[i,j],3)
        dR2 = size(M[ir,j],3)
        BgFR[i,j] = rand(Float64, D1, dR1, dR2, D2)
    end
    return BgFR
end

"""
    FRm = FRmap(ARi, ARip, Mi, FR, J)

ARip means ARᵢ₊₁
```
 ──┐          ...  ─── ARᵢⱼ₋₁  ── ARᵢⱼ  ──┐ 
   │                    │          │      │ 
 ──│          ... ──── Mᵢⱼ₋₁  ──  Mᵢⱼ   ──│
  BgFRm   =             │          │     BgFRm
 ──│          ... ──── Mᵢ₊₁ⱼ₋₁ ── Mᵢ₊₁ⱼ ──│
   │                    │          │      │     
 ──┘          ...  ─ ARᵢ₊₂ⱼ₋₁ ─── ARᵢ₊₂ⱼ──┘ 
```
"""
function BgFRmap(ARi, ARip, Mi, Mip, BgFR, J)
    Nj = size(ARi,1)
    BgFRm = copy(BgFR)
    for j=1:Nj
        jr = J-(j-1) + (J-(j-1) < 1)*Nj
        BgFRm = ein"fghi,def,ckge,bjhk,aji -> dcba"(BgFRm,ARi[jr],Mi[jr],Mip[jr],conj(ARip[jr]))
    end
    return BgFRm
end

"""
    λR, BgFR = bigrightenv(AR, M, BgFR = BgFRint(AR,M); kwargs...)

Compute the right environment tensor for MPS A and MPO M, by finding the left fixed point
of AR - M - M - conj(AR) contracted along the physical dimension.
```
     ──┐          ...  ─── ARᵢⱼ₋₁  ── ARᵢⱼ  ──┐ 
       │                    │          │      │ 
     ──│          ... ──── Mᵢⱼ₋₁  ──  Mᵢⱼ   ──│
λRᵢⱼ BgFRᵢⱼ   =             │          │     BgFRᵢⱼ
     ──│          ... ──── Mᵢ₊₁ⱼ₋₁ ── Mᵢ₊₁ⱼ ──│
       │                    │          │      │     
     ──┘          ...  ─ ARᵢ₊₂ⱼ₋₁ ─── ARᵢ₊₂ⱼ──┘ 
```
"""
bigrightenv(AR, M, BgFR = BgFRint(AR,M); kwargs...) = bigrightenv!(AR, M, copy(BgFR); kwargs...)
function bigrightenv!(AR, M, BgFR; kwargs...)
    Ni,Nj = size(AR)
    λR = zeros(Ni,Nj)
    Threads.@threads for k = 1:Ni*Nj
        i,j = ktoij(k, Ni, Nj)
        ir = i + 1 - Ni * (i==Ni)
        # irr = i + 2 - Ni * (i + 2 > Ni) # modified for 2x2
        λRs, BgFR1s, _= eigsolve(X->BgFRmap(AR[i,:], AR[ir,:], M[i,:], M[ir,:], X, j), BgFR[i,j], 1, :LM; ishermitian = false, kwargs...)
        if length(λRs) > 1 && norm(abs(λRs[1]) - abs(λRs[2])) < 1e-12
            @show λRs
            if real(λRs[1]) > 0
                BgFR[i,j] = real(BgFR1s[1])
                λR[i,j] = real(λRs[1])
            else
                BgFR[i,j] = real(BgFR1s[2])
                λR[i,j] = real(λRs[2])
            end
        else
            BgFR[i,j] = real(BgFR1s[1])
            λR[i,j] = real(λRs[1])
        end
    end
    return λR, BgFR
end
