# Sourced by deploy-containerapp.sh: read one API key with hidden input.
#
# read_key NAME PREFIX prints the cleaned key on stdout and its messages on
# stderr, never the key itself. It removes common paste mistakes (surrounding
# quotes or spaces, a Windows line ending, a copied "NAME=" or
# "export NAME=" prefix), then checks that the key starts with PREFIX. A
# rejected paste is asked for again, up to three times.
# CHECK_KEY_PREFIXES=0 skips the prefix check for keys in an older format.

read_key() {
  local name="$1" prefix="$2" value _attempt
  for _attempt in 1 2 3; do
    value=""
    read -r -s -p "$name (hidden): " value || true
    echo >&2
    value="$(printf '%s' "$value" | tr -d '\r' | sed -E \
      -e 's/^[[:space:]]+//; s/[[:space:]]+$//' \
      -e "s/^(export[[:space:]]+)?${name}[[:space:]]*=[[:space:]]*//" \
      -e "s/^[\"']//; s/[\"']\$//")"
    if [[ -z "$value" ]]; then
      echo "Nothing was pasted. Paste the key, then press Enter." >&2
    elif [[ "$value" =~ [[:space:]] ]]; then
      echo "The pasted text contains spaces or several lines. Copy only the key." >&2
    elif [[ "${CHECK_KEY_PREFIXES:-1}" != 0 && "$value" != "$prefix"* ]]; then
      echo "$name keys start with '$prefix', but the pasted text does not." \
        "Copy only the part after '$name='." >&2
    else
      if [[ "$value" == "$prefix"* ]]; then
        echo "$name accepted: ${#value} characters, starts with '$prefix'." >&2
      else
        echo "$name accepted: ${#value} characters (prefix check skipped)." >&2
      fi
      printf '%s' "$value"
      return 0
    fi
  done
  echo "$name was not accepted after 3 attempts; nothing was changed." >&2
  return 1
}
