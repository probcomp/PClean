struct ImportanceResampling <: Algorithm
  particles :: Int
end

export ImportanceResampling

function infer_row_trace(alg::ImportanceResampling, row::NamedTuple, dbm::DBModel)
  obs = choicemap()
  for k in keys(dbm.alignment)
    if k in keys(row) && !ismissing(row[k])
      obs[dbm.alignment[k]] = row[k]
    end
  end
  final_trace, _ = importance_resampling(dbm.model, Tuple([nothing; dbm.params]), obs, alg.particles)
  return final_trace
end
