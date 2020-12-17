struct InferenceConfig
  num_iters        :: Int
  num_particles    :: Int
  use_dd_proposals :: Bool # data-driven proposals
  use_lo_sweeps    :: Bool # latent-object PGibbs sweeps
  use_mh_instead_of_pg :: Bool # MH accept/reject rule
  rejuv_frequency  :: Int # frequency of *parameter* rejuvenation moves
  reporting_frequency :: Int

  function InferenceConfig(num_iters, num_particles; use_dd_proposals=true, use_lo_sweeps=true, use_mh_instead_of_pg=false, rejuv_frequency=50, reporting_frequency=100)
    if use_mh_instead_of_pg
      num_particles = 2
    end
    return new(num_iters, num_particles, use_dd_proposals, use_lo_sweeps, use_mh_instead_of_pg, rejuv_frequency, reporting_frequency)
  end
end
