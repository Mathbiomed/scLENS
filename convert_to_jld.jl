using DataFrames
using ArgParse
using MatrixMarket
using CSV
using JLD2
using CodecLz4:LZ4FrameCompressor as lz4
using GZip
using SparseArrays
function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--out_dir", "-d"
            help = "Specify the directory where jld2 files will be saved."
            arg_type = String
            default = tempname()
        "--out_name", "-n"
            help = "Name of jldfile"
            arg_type = String
            default = "tmp_1.jld2"
        "arg1"
            help = "Provide the path to the input file, which should be a CSV or JLD2 file. The file must contain a matrix with rows representing cells and columns representing genes."
            required = true
    end
    return parse_args(s)
end


function main()
    @show parsed_args = parse_commandline()
    println("Parsed args:")
    for (arg,val) in parsed_args
        print("  $arg  =>  ")
        show(val)
        println()
    end

    p_dir = parsed_args["arg1"]
    # p_dir = raw"C:\Users\kimma\Downloads\CytAssist_FreshFrozen_Mouse_Brain_Post_Xenium_Rep1_filtered_feature_bc_matrix\filtered_feature_bc_matrix"
    o_dir = parsed_args["out_dir"]
    # o_dir = tempname()
    f_name = parsed_args["out_name"]
    # f_name = "tmp_1.jld2"
    M = try
        tmp_f = joinpath(p_dir,"matrix.mtx.gz")
        f_obj_ = GZip.open(tmp_f)
        tmp_obj = readlines(f_obj_)
        n = length(tmp_obj)-2
        a_tmp = [parse.(Int,split(s," ")) for s in tmp_obj[3:end]]
        I = [s[1] for s in a_tmp[2:end]]
        J = [s[2] for s in a_tmp[2:end]]
        K = [s[3] for s in a_tmp[2:end]]
        GZip.close(f_obj_)
        sparse(I,J,K,a_tmp[1][1],a_tmp[1][2])
    catch
        mmread(joinpath(p_dir,"matrix.mtx"))
    end
    cells_ = try
        values(CSV.read(joinpath((p_dir,"barcodes.tsv.gz")),DataFrame,header=false)[!,1])
    catch
        values(CSV.read(joinpath((p_dir,"barcodes.tsv")),DataFrame,header=false)[!,1])
    end
    # gene_ = values(CSV.read(joinpath(tmp_dir,"features.tsv.gz"),DataFrame,header=false)[!,2])
    gene_ = try
        values(CSV.read(joinpath(p_dir,"features.tsv.gz"),DataFrame,header=false)[!,2])
    catch
        values(CSV.read(joinpath(p_dir,"features.tsv"),DataFrame,header=false)[!,2])
    end
    ndf = DataFrame(M',gene_,makeunique=true)
    insertcols!(ndf,1,:cell => cells_)
    onn = joinpath(o_dir,f_name)
    JLD2.save(onn,Dict("data" => ndf);compress=lz4())
    println("$onn is saved!")
end
main()