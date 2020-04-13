/*
Fetch the value associated with a given key from an array of key/value structs.
Because map types aren't available in BigQuery, we model maps as arrays
of structs instead, and this function provides map-like access to such fields.
*/
CREATE OR REPLACE FUNCTION udf.get_key(map ANY TYPE, k ANY TYPE) AS (
  (SELECT key_value.value FROM UNNEST(map) AS key_value WHERE key_value.key = k LIMIT 1)
);
