source(file.path("/projects/f_ps848_1/zhaobin/causal-saliency/results", "utils.R"))

library("arrow")
library("ggplot2")
library("tidyverse")
library("rlang")
library("glue")
library("testthat")

# data_path <- "/projects/f_ps848_1/zhaobin/causal-saliency/data"
data_path <- "/projects/f_ps848_1/zhaobin/causal-saliency/data_nonasym"
graph_path <- "/projects/f_ps848_1/zhaobin/causal-saliency/graphs"
dir.create(graph_path, showWarnings = FALSE)

# files <- dir(data_path, pattern = "*.feather") # get file names
files <- Sys.glob(file.path(data_path, "*asymptotic_sim=False*.feather")) # get file names
print(files)

# Ref: https://clauswilke.com/blog/2016/06/13/reading-and-combining-many-tidy-data-files-in-r/
data <- files %>%
  # read in all the files, appending the path before the filename
  map_dfr(~read_feather(.))

glimpse(data)

data <- data %>%
  select(!c(asym_coefs, mse, rel_mse))

data %>%
  summarise(n_distinct(img_path))

data %>%
  group_by(num_segs) %>%
  summarise(n())


# estimate the ground-truth coefs and get rel_mse
data <- data %>%
  group_by(img_path, sp_name, num_segs, sim_name) %>% # whatever pert_name, est_name is, we still estimate the ground truth
  mutate(rel_mse = get_wt_rel_mse(coefs, n_sam)) %>%
  ungroup()

# get pairwise correlation within a simulation class
data <- data %>%
  group_by(img_path, sp_name, num_segs, pert_name, sim_name, est_name, n_sam) %>%
  mutate(pair_corr = get_mean_pair_corr(coefs)) %>%
  ungroup()

df <- data %>%
  group_by(img_path, sp_name, pert_name, sim_name, est_name, n_sam) %>%
  summarise(mean_rel_mse = mean(rel_mse), sd_rel_mse = sd(rel_mse)) %>%
  group_by(sp_name, pert_name, sim_name, est_name, n_sam) %>%
  summarise(qs = quantile(mean_rel_mse, c(0.05, 0.25, 0.75, 0.95)), prob = c(0.05, 0.25, 0.75, 0.95))

data %>%
  group_by(sp_name, num_segs, pert_name, sim_name, est_name, n_sam, img_path) %>%
  summarise(mean_pair_corr = mean(pair_corr), sd_pair_corr = sd(pair_corr)) %>%
  summarise(qs = quantile(mean_pair_corr, c(0.05, 0.25, 0.75, 0.95)), prob = c(0.05, 0.25, 0.75, 0.95), .groups = "drop_last")


# y_colnames <- c("rel_mse", "pair_corr", "dauc", "iauc")
y_colnames <- c("rel_mse")
for (y in y_colnames) {
  print(y)
  print(data %>%
          group_by(img_path, sp_name, pert_name, sim_name, est_name, n_sam) %>%
          summarise("mean_{y}" := mean(!!sym(y))) %>%
          ggplot(aes(!!sym(glue("mean_{y}")), colour = est_name, linetype = pert_name)) +
          geom_freqpoly(bins = 100) +
          facet_grid(cols = vars(sp_name), rows = vars(n_sam)) +
          scale_y_log10() +
          scale_x_log10() +
          labs(title = y))
  ggsave(file.path(graph_path, glue('{y}_hist.png')))
}


y_colnames <- c("rel_mse", "pair_corr", "dauc", "iauc")
# y_colnames <- c("pair_corr")
for (y in y_colnames) {
  print(y)
  print(ggplot(data = data, aes(x = n_sam, y = !!sym(y), color = est_name, linetype = pert_name)) +
          stat_summary(fun.data = "mean_se") +
          # geom_smooth(method = 'lm', formula = y ~ I(1 / x)) +
          geom_smooth(method = 'lm', formula = y ~ x) +
          facet_grid(cols = vars(sp_name), rows = vars(sim_name)) +
          labs(title = y) +
          scale_y_log10() +
          scale_x_log10())
  ggsave(file.path(graph_path, glue('{y}_stats.png')))
}
