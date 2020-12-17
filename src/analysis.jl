using Dates

is_saveable(::RandomChoiceNode) = true
is_saveable(::JuliaNode) = true
is_saveable(::ForeignKeyNode) = true
is_saveable(::PCleanNode) = false

function save_tables(dir, trace)
    for (class, table) in trace.tables 
        tbldf = CSV.DataFrame(; id = collect(keys(trace.rows)), [k => [row[i] for row in values(trace.rows)] for (k, i) in trace.model.classes[class].names if !occursin("#", string(k)) && is_saveable(trace.model.classes[class].nodes[i])]...)
        CSV.write("$dir/inferred_$class.csv", tbldf)
    end
end

function save_results(dir, name, trace, observed_datasets, timestamp=true)
  dir = timestamp ? "$dir/$name-$(now())" : "$dir/$name"
  mkpath(dir)

  for (n, dataset) in observed_datasets
    query = dataset.query
    class = query.class
    table_trace = trace.tables[class]

    tbldf = CSV.DataFrame(; [k => haskey(query.cleanmap, k) ?
                                [table_trace.rows[i][query.cleanmap[k]]
                                    for i in sort(collect(keys(table_trace.rows)))] :
                                table[!, k]
                            for k in names(dataset.data)]...)
    CSV.write("$dir/reconstructed_$class.csv", tbldf)
  end
  save_tables(dir, trace)
end


function evaluate_accuracy(dirty_data, clean_data, table, query)
  total_errors = 0
  total_changed = 0 # not including imputed
  total_cleaned = 0 # correct repairs; total_changed - total_cleaned gives incorrect repairs
  total_imputed = 0
  total_imputed_correctly = 0

  n_rows = length(table.rows)
  pcleaned_rows = [table.rows[i] for i=1:n_rows]
  cleanmap = query.cleanmap
  for (dirty, clean, ours) in zip(eachrow(dirty_data), eachrow(clean_data), pcleaned_rows)
    for colname in names(clean_data)
      if !haskey(dirty, colname)
        continue
      end

      # Currently, we don't count missing values as errors.
      if ismissing(dirty[colname])
        if haskey(cleanmap, colname) && !ismissing(clean[colname])
          total_imputed += 1
          if ours[cleanmap[colname]] == clean[colname]
            total_imputed_correctly += 1
          end
        end
        continue
      end

      if dirty[colname] != clean[colname]
        total_errors += 1
      end

      if haskey(cleanmap, colname)
        our_version = ours[cleanmap[colname]]
        if our_version != dirty[colname]
          total_changed += 1
          if our_version == clean[colname]
            total_cleaned += 1
          else
            println("Changed: $(dirty[colname]) -> $our_version instead of $(clean[colname])")
          end
        else
          if dirty[colname] != clean[colname]
            println("Left unchanged: $(dirty[colname]) (should be $(clean[colname]))")
          end
        end
      end
    end
  end
  precision = (total_cleaned + total_imputed_correctly) / (total_changed + total_imputed)
  recall    = (total_cleaned + total_imputed_correctly) / (total_errors + total_imputed)
  f1 = 2.0/(1/precision + 1/recall)
  return (f1=f1, errors=total_errors, changed=total_changed, cleaned=total_cleaned,
          precision = precision, recall = recall,
          imputed = total_imputed, correctly_imputed = total_imputed_correctly)
end



function evaluate_accuracy_up_to(dirty_data, clean_data, table, query, N)
  total_errors = 0
  total_changed = 0 # not including imputed
  total_cleaned = 0 # correct repairs; total_changed - total_cleaned gives incorrect repairs
  total_missing = 0
  total_imputed = 0
  total_imputed_correctly = 0

  n_rows = length(eachrow(dirty_data))
  pcleaned_rows = [i <= N ? table.rows[i] : nothing for i=1:n_rows]
  cleanmap = query.cleanmap

  for (dirty, clean, ours) in zip(eachrow(dirty_data), eachrow(clean_data), pcleaned_rows)
    for colname in names(clean_data)
      if !haskey(dirty, colname)
        continue
      end

      # Currently, we don't count missing values as errors.
      if ismissing(dirty[colname])
        if haskey(cleanmap, colname) && !ismissing(clean[colname])
          if !isnothing(ours)
            total_imputed += 1
          end
          total_missing += 1
          if !isnothing(ours) && ours[cleanmap[colname]] == clean[colname]
            total_imputed_correctly += 1
          end
        end
        continue
      end

      if dirty[colname] != clean[colname]
        total_errors += 1
      end

      if haskey(cleanmap, colname) && !isnothing(ours)
        our_version = ours[cleanmap[colname]]
        if our_version != dirty[colname]
          total_changed += 1
          if our_version == clean[colname]
            total_cleaned += 1
          else
           # println("Changed: $(dirty[colname]) -> $our_version instead of $(clean[colname])")
          end
        else
          if dirty[colname] != clean[colname]
           # println("Left unchanged: $(dirty[colname]) (should be $(clean[colname]))")
          end
        end
      end
    end
  end
  precision = (total_cleaned + total_imputed_correctly) / (total_changed + total_imputed)
  recall    = (total_cleaned + total_imputed_correctly) / (total_errors + total_missing)
  f1 = 2.0/(1/precision + 1/recall)
  return (f1=f1, errors=total_errors, changed=total_changed, cleaned=total_cleaned,
          precision = precision, recall = recall,
          imputed = total_imputed, correctly_imputed = total_imputed_correctly)
end



export save_results, evaluate_accuracy, evaluate_accuracy_up_to
