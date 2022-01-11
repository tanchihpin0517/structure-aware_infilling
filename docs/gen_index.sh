PROJECT_DIR="/screamlab/home/tanch/structural_expansion"
cp -r $PROJECT_DIR/gen_midi/* $PROJECT_DIR/docs/assets/songs
python gen_index.py \
    -songs_dir "$PROJECT_DIR/docs/assets/songs" \
    -templates_dir "$PROJECT_DIR/docs/templates" \
    -index_file "$PROJECT_DIR/docs/index.html"
