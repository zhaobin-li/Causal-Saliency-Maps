library("arrow")
library("ggplot2")
library("tidyverse")
library("rlang")
library("glue")
library("testthat")

data_path <- "/projects/f_ps848_1/zhaobin/causal-saliency/data"
graph_path <- "/projects/f_ps848_1/zhaobin/causal-saliency/graphs"
dir.create(graph_path, showWarnings = FALSE)

# data_path <- "/Users/zhaobinli/Sync/TrustedProjects/causal-saliency/data"
# df <- read_feather(file.path(data_path, "mse_asymptotic.feather"))

files <- dir(data_path, pattern = "*.feather") # get file names

# Ref: https://clauswilke.com/blog/2016/06/13/reading-and-combining-many-tidy-data-files-in-r/
data <- files %>%
  # read in all the files, appending the path before the filename
  map_dfr(~read_feather(file.path(data_path, .)))

glimpse(data)

data <- data %>%
  filter(pert_name != 'perm') %>%
  filter(n_sam >= 100)

data <- data %>% mutate(rel_n_sam = n_sam / (2^num_segs))

# y_colnames <- c("rel_mse", "dauc", "iauc")
y_colnames <- c("rel_mse")
for (y in y_colnames) {
  print(y)
  print(ggplot(data = data, aes(!!sym(y), colour = est_name)) +
          geom_freqpoly(bins = 100) +
          scale_y_log10() +
          scale_x_log10() +
          facet_grid(rows = vars(sim_name), cols = vars(pert_name)) +
          labs(title = y))
  ggsave(file.path(graph_path, glue('{y}_hist.png')))
}


# y_colnames <- c("rel_mse", "dauc", "iauc")
y_colnames <- c("dauc")
for (y in y_colnames) {
  print(y)
  print(ggplot(data = data, aes(x = rel_n_sam, y = !!sym(y), color = est_name, linetype = pert_name)) +
          stat_summary_bin(bins = 5, fun.data = "mean_se") +
          geom_smooth(method = 'lm', formula = y ~ I(1 / x)) +
          facet_grid(cols = vars(sp_name), rows = vars(sim_name)) +
          labs(title = y))
  ggsave(file.path(graph_path, glue('{y}_stats.png')))
}


data <- read_feather(file.path(data_path, "rel_mse_nonasymptotic_volta001, cuda:0_y_num=3, x_num=3, num_imgs=1, num_reps=50.feather"))
glimpse(data)
data %>% select(sp_name, coefs)
data %>%
  group_by(n_img, sp_name, sim_name)

ll <- list(c(1, 2), c(3, 4))
print(ll)

mean_ll <- c(2, 3)

get_mean <- function(.) {
  print(.)
}

expect_equal(get_mean(ll), mean_ll)

data <- data %>%
  select(sp_name, coefs) %>%
  slice_head(n = 5)

data <- data %>%
  select(coefs) %>%
  unnest_wider(coefs) %>%
  summarise(across(everything(), mean)) %>%
  nest(mean_coefs = everything()) %>%
  bind_cols(data) %>%
  rowwise() %>%
  mutate(mse = get_eucl_dist(mean_coefs, coefs))

get_eucl_dist <- function(x, y) {
  norm((x - y), type = "2")
}

data %>%
  group_by(n_img, sp_name, sim_name) %>%
  select(coefs) %>%
  unnest_wider(coefs, names_sep = "_") %>%
  summarise(across(contains("coef_"), mean)) %>%
  nest(mean_coefs = contains("coef_")) %>%
  ungroup()
# bind_cols(data) %>%
rowwise() %>%
  mutate(mse = get_eucl_dist(mean_coefs, coefs))

# pull(mean_coefs)
#
#   mutate(mean_coefs = get_mean(coefs))
# df$mean_coefs
# df$coefs
