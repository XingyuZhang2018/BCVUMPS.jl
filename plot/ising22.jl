using BCVUMPS
using Plots
using JLD2
using FileIO
using Random

# 2x2 Ising
begin
    model = Ising22(2)
    magplot = plot()
    xlabel!("β")
    ylabel!("mag")
    eneplot = plot()
    xlabel!("β")
    ylabel!("ene")
    for D = [2,4,8]
        mag = []
        ene = []
        for β = 0.26:0.005:0.36
            printstyled("BCVUMPS D = $(D), β = $(β) \n"; bold=true, color=:red)
            
            env = bcvumps_env(model, β, D)
            mag = [mag;magnetisation(env, model, β)]
            ene = [ene;energy(env, model, β)]
            chkp_file = "./data/$(model)_β$(round(β+0.005,digits=4))_D$(D).jld2"
            if isfile(chkp_file) == false
                save(chkp_file, "env", env)
            end
        end
        β = 0.26:0.005:0.36
        plot!(magplot,β,mag,seriestype = :scatter,title = "magnetisation", label = "BCVUMPS D = $(D)", lw = 3)
        plot!(eneplot,β,ene,seriestype = :scatter,title = "energy", label = "BCVUMPS D = $(D)", lw = 3)
    end
    for L = [8,12,16]
        mcmag = []
        mcene = []
        for β = 0.26:0.005:0.36
            printstyled("MCMC L = $(L) β = $(β) \n"; bold=true, color=:red)
            mag22,ene22 = MCMC(model,L,β,10000,100000)
            mcmag = [mcmag;mag22]
            mcene = [mcene;ene22]
        end
        β = 0.26:0.005:0.36
        plot!(magplot,β,mcmag,label = "MCMC L = $(L)", lw = 2)
        plot!(eneplot,β,mcene,label = "MCMC L = $(L)", lw = 2)
    end
    t = plot(magplot, eneplot, layout = (1, 2), legend=:bottomleft)
    savefig(t,"./plot/$(model).svg")
end