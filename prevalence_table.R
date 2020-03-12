library(data.table)
`%+=%` = function(e1,e2) eval.parent(substitute(e1 <- e1 + e2))

files <- list.files(path="/scratch/derickmath/interpret/ECG-interpretation/output_result/means/", pattern="ecg*", full.names=TRUE, recursive=FALSE);
threshold = c(0.15, 0.1, 0.08, 0.33, 0.27, 0.20, 1);
p_score = c(0,0,0,0,0,0,0)
t_score = c(0,0,0,0,0,0,0)
pr_score = c(0,0,0,0,0,0,0)
qt_score = c(0,0,0,0,0,0,0)
AV_rate_score = c(0,0,0,0,0,0,0)
st_score = c(0,0,0,0,0,0,0)
qrs_score = c(0,0,0,0,0,0,0)
axis_score = c(0,0,0,0,0,0,0)
rhythm_score = c(0,0,0,0,0,0,0)
random15_score = c(0,0,0,0,0,0,0)
random30_score = c(0,0,0,0,0,0,0)
random50_score = c(0,0,0,0,0,0,0)
random_score = c(0,0,0,0,0,0,0)
random2_score = c(0,0,0,0,0,0,0)
original_score = c(0,0,0,0,0,0,0)

for(i in 1:length(files)){
  source(files[i]);
  data_file = data.frame(p, t, pr, qt, AV_rate, st, qrs, axis, rhythm, random15, random30, random50, random, random2);
  AV_rate_score %+=% as.integer(abs(AV_rate)>=threshold);
  p_score %+=% as.integer(abs(p)>=threshold);
  t_score %+=% as.integer(abs(t)>=threshold);
  pr_score %+=% as.integer(abs(pr)>=threshold);
  qt_score %+=% as.integer(abs(qt)>=threshold);
  st_score %+=% as.integer(abs(st)>=threshold);
  qrs_score %+=% as.integer(abs(qrs)>=threshold);
  axis_score %+=% as.integer(abs(axis)>=threshold);
  rhythm_score %+=% as.integer(abs(rhythm)>=threshold);
  random15_score %+=% as.integer(abs(random15)>=threshold);
  random30_score %+=% as.integer(abs(random30)>=threshold);
  random50_score %+=% as.integer(abs(random50)>=threshold);
  random_score %+=% as.integer(abs(random)>=threshold);
  random2_score %+=% as.integer(abs(random2)>=threshold);
  
  original_score %+=% as.integer(abs(original[2:8])>=threshold)
}
data_file = data.frame(p_score, t_score, pr_score, qt_score, AV_rate_score, st_score, qrs_score, axis_score,
                       rhythm_score, random15_score, random30_score, random50_score, random_score, random2_score,
                       original_score);

write.csv(data_file, row.names = FALSE, file = "~/table_pkdd.csv")