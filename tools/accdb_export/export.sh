#!/usr/bin/env sh
# Re-create data/raw/nabih-magneto/derived/*.tsv from the original Access database with Jackcess 4.0.5.
# Needs a JDK (javac, java) and curl. Jars come from Maven Central and are checked against the SHA-1 values in EXPORT.json.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
work=$(mktemp -d)
for jar in com/healthmarketscience/jackcess/jackcess/4.0.5/jackcess-4.0.5.jar:78e71fb55d742715b7a2dc8e0a6c2dce218c8e17 \
           org/apache/commons/commons-lang3/3.14.0/commons-lang3-3.14.0.jar:1ed471194b02f2c6cb734a0cd6f6f107c673afae \
           commons-logging/commons-logging/1.3.0/commons-logging-1.3.0.jar:f5e064b541f5c5fbc5e4fb49c4e8cd4eabb3afd6; do
  rel=${jar%%:*}; sha=${jar##*:}; name=$(basename "$rel")
  curl -sfL -o "$work/$name" "https://repo1.maven.org/maven2/$rel"
  [ "$(shasum -a 1 "$work/$name" | cut -c1-40)" = "$sha" ] || { echo "sha1 mismatch: $name" >&2; exit 1; }
done
javac -d "$work" -cp "$work/*" "$here/Dump.java"
out=${1:-"$work/out"}; mkdir -p "$out"
java -cp "$work:$work/*" Dump "$root/data/raw/nabih-magneto/files/Core loss database_v2.accdb" "$out" > /dev/null
echo "exported to $out"
