import CSV
import JuliaDB

abstract type Algorithm end

"""
   loadcsv(f::String)
Load a CSV file as a table from disk.
"""
loadcsv(f::String) = JuliaDB.table(CSV.File(f))

function clean(alg::Algorithm, dbm::DBModel, tbl)
  map(tbl) do row
    final_trace = infer_row_trace(alg, row, dbm)
    merge(row, get_retval(final_trace)[1])
  end
end

function fit!(alg::Algorithm, dbm::DBModel, tbl; iters=1)
  for _=1:iters
    complete_data = map(tbl) do row
      final_trace = infer_row_trace(alg, row, dbm)
      merge(row, get_retval(final_trace)...)
    end

    for i=1:length(dbm.params)
      dbm.params[i] = update_param(dbm.params[i], complete_data)
    end
  end
end

export loadcsv
export clean
export fit!
