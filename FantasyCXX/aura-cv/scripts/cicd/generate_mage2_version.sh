#!/bin/bash

# Recursively search for files with the specified extension and generate an aligned JSON file
generate_json() {
    if [ $# -lt 8 ]; then
        echo "param < 8"
        exit 1
    fi

    local search_path=$1        # Path to search
    local file_extension=$2     # File extension
    local output_path=$3        # Path to save the output file
    local output_filename=$4    # Output file name
    local get_aura2_script=$5   # get_aura2.py
    local zip_name=$6           # zip name

    # FDS parameters
    local fds_region=$7         # FDS region
    local fds_bucket=$8         # FDS bucket name

    # Recursively search for files with the specified extension
    files=$(find "$search_path" -type f -name "*.$file_extension")

    # Initialize the output file
    output_file="$output_path/$output_filename"
    temp_file=$(mktemp)
    echo "{" > "$temp_file"

    # Process each file
    for file in $files; do
        # Get the file name (remove path and extension)
        filename=$(basename "$file" ".$file_extension")
        
        # Construct the JSON line
        echo "    \"$filename\"  : 0," >> "$temp_file"
    done

    # Calculate the maximum key length
    max_key_length=0
    while IFS= read -r line; do
        if [[ $line =~ \"(.*)\"\ *:\ 0, ]]; then
            key_length=${#BASH_REMATCH[1]}
            if (( key_length > max_key_length )); then
                max_key_length=$key_length
            fi
        fi
    done < "$temp_file"

    # Generate the aligned file
    echo "{" > "$output_file"
    while IFS= read -r line; do
        if [[ $line =~ \"(.*)\"\ *:\ 0, ]]; then
            key=${BASH_REMATCH[1]}
            spaces=$((max_key_length - ${#key}))
            printf "    \"%s\"%${spaces}s : 0,\n" "$key" >> "$output_file"
        fi
    done < "$temp_file"

    # Remove the last comma and close the JSON format
    truncate -s-2 "$output_file"
    echo "" >> "$output_file"
    echo "}" >> "$output_file"

    # Remove the temporary file
    rm "$temp_file"

    # change the rigion and bucket name of get_aura2.py
    sed -i "s/FDS_REGION/$fds_region/g"      "$output_path"/"$get_aura2_script"
    sed -i "s/FDS_BUCKET_NAME/$fds_bucket/g" "$output_path"/"$get_aura2_script"

    # zip
    zip -j "$output_path"/"$zip_name".zip "$output_file" "$output_path"/"$get_aura2_script" > /dev/null
    rm  "$output_file"
    rm  "$output_path"/"$get_aura2_script"
}

# Call the function
# Parameter 1: Path to search
# Parameter 2: File extension (e.g., zip)
# Parameter 3: Path to save the output file
# Parameter 4: Output file name
# Parameter 5: get_aura2.py
# Parameter 6: zip name
# such as: generate_json "." "zip" "." "output.json" "get_aura2.py" "get_aura2"
generate_json "$@"


