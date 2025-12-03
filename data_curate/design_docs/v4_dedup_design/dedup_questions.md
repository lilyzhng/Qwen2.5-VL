# Questions
[Q1]. why sampling 8 frames / 3000 frames is not diverse?
let's say each slice is 15 seconds of data and each slice is 300 frames. That means that for 10 slices, it is 150 seconds and 3,000 frames. If we're only sampling from 8 frames of a total 10 slices, I think it's lots of diversity already.
could you help me understand why it would be "the same slices will be selected together repeatedly."? Even selecting 8 / 3000 should result in very different data. Maybe I missed something. 

[Q2]. does the current sampling of 8 frames happen on the row level?
existing: 10 rows per file, each row = 1 slice = 300 frames = 15 seconds. When training: sample a row, then take 8 frames from that row (is this true?)
new: 10 × 300 / 10 = 300 rows per file, each row = 0.5 seconds = 10 frames. When training: sample a row, then take 8 frames from that row

[Q3]. I am wondering why restructure the entire dataset instead of changing the sampling strategy?
The data is already bundled with 10 slices per file (3,000 frames total)
Instead of sampling from a single row, couldn't the data loader sample frames across all 10 rows within the bundled file?

[Q4] why do we choose “2000 rows per parquet file” as the target?


10 rows,  each row of 150 frames
10*150/10 rows = 150 rows, each row of 10 frames