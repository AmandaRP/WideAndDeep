

user_data <- jsonlite::stream_in(gzcon(url("http://deepyeti.ucsd.edu/jmcauley/datasets/googlelocal/users.clean.json.gz")))

tmp <- tempfile()
download.file("http://deepyeti.ucsd.edu/jmcauley/datasets/googlelocal/users.clean.json.gz", tmp)
user_data <- jsonlite::stream_in(gzfile(tmp))
