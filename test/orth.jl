@testset "orth  with ($Ni $Nj)" for Ni in [1,2,3], Nj in [1,2,3]
    D = 10
    A = Array{Array,2}(undef, Ni, Nj)
    for i = 1:Ni,j = 1:Nj
        Random.seed!(1234+i+j)
        A[i,j] = randn(D, 2, D) + im*randn(D, 2, D)
    end
    AL, L ,λL = BCVUMPS.leftorth(A)
    R, AR,λR = BCVUMPS.rightorth(A,L)
    for i = 1:Ni,j = 1:Nj
        @tensor AL2[b, c] := AL[i,j][a,s,b]*conj(AL[i,j][a,s,c])
        @test Array(AL2) ≈ I(D)
        @tensor AR2[b, c] := AR[i,j][b,s,a]*conj(AR[i,j][c,s,a])
        @test Array(AR2) ≈ I(D)
        
        @tensor ALL[a,s,c] := AL[i,j][a,s,b]*L[i,j][b,c]
        @tensor LA[a,s,c] := L[i,j][a,b]*A[i,j][b,s,c]
        @test LA ≈ ALL * λL[i,j]
        
        @tensor RAR[a,s,c] := R[i,j][a,b]*AR[i,j][b,s,c]
        @tensor A_R[a,s,c] := A[i,j][a,s,b]*R[i,j][b,c]
        @test A_R ≈ RAR * λR[i,j]
#         L_2 = ρmap(L[i,j]' * L[i,j], A[i,:], j)
#         @test std(L_2./(L[i,j]' * L[i,j])) ≈ 0 atol=1e-10
#         @test AL[1,1] ≈ AL[1,2] ≈ AL[2,1] ≈ AL[2,2]
#         @test AR[1,1] ≈ AR[1,2] ≈ AR[2,1] ≈ AR[2,2]
    end
end