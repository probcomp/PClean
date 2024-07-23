function attribute_extractors(model::PClean.PCleanModel)
    physician_attributes = Dict(
      "npi"=>PClean.resolve_dot_expression(model, :Obs, :(p.npi)),
      "first"=>PClean.resolve_dot_expression(model, :Obs, :(p.first)),
      "last"=>PClean.resolve_dot_expression(model, :Obs, :(p.last)),
      "degree"=>PClean.resolve_dot_expression(model, :Obs, :(p.degree)),
      "speciality"=>PClean.resolve_dot_expression(model, :Obs, :(p.specialty)),
      "school" => PClean.resolve_dot_expression(model, :Obs, :(p.school.name))
    )

    business_attributes = Dict(
      "legal_name" => PClean.resolve_dot_expression(model, :Obs, :(a.legal_name)),
      "addr" => PClean.resolve_dot_expression(model, :Obs, :(a.addr)),
      "addr2" => PClean.resolve_dot_expression(model, :Obs, :(a.addr2)),
      "zip" => PClean.resolve_dot_expression(model, :Obs, :(a.zip)),
      "city" => PClean.resolve_dot_expression(model, :Obs, :(a.city.name)),
    )

    function attributes(row)
      physician_attr = Dict(attribute=>row[id] for (attribute, id) in physician_attributes)
      business_attr = Dict(attribute=>row[id] for (attribute, id) in business_attributes)
      physician_id = row[PClean.resolve_dot_expression(model, :Obs, :p)]
      business_id = row[PClean.resolve_dot_expression(model, :Obs, :a)]
      return (physician_id, business_id), physician_attr, business_attr
    end

    return attributes
end