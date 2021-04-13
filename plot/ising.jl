using BCVUMPS
using Plots
using JLD2
using FileIO

# 1x1 Ising magnetisation
begin
    magplot = plot()
    model = Ising()
    for D = [2,4,8,16]
        mag = []
        for β = 0.41:0.0025:0.48
            printstyled("D = $(D), β = $(β) \n"; bold=true, color=:red)
            
            env = bcvumps_env(model, β,D)
            mag = [mag;magnetisation(env, model, β)]
            chkp_file = "./data/$(model)_β$(round(β+0.0025,digits=4))_D$(D).jld2"
            if isfile(chkp_file) == false
                save(chkp_file, "env", env)
            end
        end
        β = 0.41:0.0025:0.48
        magplot = plot!(β,mag,seriestype = :scatter,title = "magnetisation", label = "BCVUMPS D = $(D)", lw = 3)
    end

    tmag = []
    for β = 0.41:0.001:0.48
        tmag = [tmag;magofβ(Ising(), β)]
    end
    β = 0.41:0.001:0.48
    magplot = plot!(β,tmag,label = "exact", lw = 2)
    xlabel!("β")
    ylabel!("mag")
    savefig(magplot,"./plot/2Disingmag.svg")
end