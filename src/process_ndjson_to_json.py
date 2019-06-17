import ndjson
import json

def process_ndjson_to_json(path_to_ndjson, path_to_json):
  '''Changing file format from newline delimited json to json with ndjson lib and saving to a file.
  Returns data in json format.'''
  with open(path_to_ndjson) as f:  
    data = ndjson.load(f)
  with open(path_to_json, "w") as f:  
    json.dump(data, f)
  return data
