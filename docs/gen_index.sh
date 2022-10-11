PROJECT_DIR=".."
#cp -r $PROJECT_DIR/demo/* $PROJECT_DIR/docs/assets/songs
python gen_index.py \
    -songs_dir "$PROJECT_DIR/docs/assets/songs" \
    -templates_dir "$PROJECT_DIR/docs/templates" \
    -index_file "$PROJECT_DIR/docs/index.html"
