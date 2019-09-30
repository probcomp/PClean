println("Loading Gen...")
using Gen
println("Loading PClean...")
using PClean

println("Loading data...")
rents = loadcsv("datasets/test.csv")

println("Defining model...")
@pclean rent_row_generator begin
  @column "A" a = normal(_, _)
  @column "B" b = bernoulli(_)
  @column "C" = normal(_[b], _)
end

println("Fitting model...")
fit!(SMC(20), rent_row_generator, rents; iters=10)

println("Cleaning...")
println(clean(SMC(20), rent_row_generator, rents))
