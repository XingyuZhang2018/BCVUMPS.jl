@testset "M ME" begin
    βc = log(1+sqrt(2))/2
    β = 0.1
    M, ME_row, ME_col,λM,λME_row,λME_col= classicalisingmpo(β; r = 1.0)
    @test M[1,1] ≈ M[1,2] ≈ M[2,1] ≈ M[2,3]
    @test ME_row[1,1] ≈ ME_row[1,2] ≈ ME_row[3,2]
    @test ME_col[2,1] ≈ ME_col[2,2] ≈ ME_col[3,2]
end