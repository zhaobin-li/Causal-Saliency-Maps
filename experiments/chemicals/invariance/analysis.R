#' ---
#' title: " Experiment Invariance Power Analysis"
#' author: "ZhaoBin Li, Scott Cheng-Hsin Yang, Tomas Folke, Patrick Shafto"
#' date: "Aug 3rd, 2022"
#' ---
#' 
# conda create -n r_chemicals -c conda-forge r-base r-jsonlite r-tidyverse r-ggplot2 r-lme4 r-afex r-furrr r-tictoc -y
library(binom)
library(viridis)

library(tictoc)
library(furrr)

library(sjPlot)
library(afex)
library(lme4)

library(testthat)
library(jsonlite)

library(ggplot2)
library(tidyverse)

MIN_AVG_RT <- 100 # ms

# Wrangle ------------------------------------------------------------

# column names in https://psiturk.readthedocs.io/en/stable/command_line.html?highlight=trialdata.csv#download-datafiles
df <-
  # read_csv("experiments/chemicals/invariance/psiturk/trialdata.csv",
  read_csv("invariance/psiturk/trialdata.csv", col_names = c("id", "trialNum", "time", "trialData")) |>
    arrange(time) |> # arrange by time
    glimpse()

# analyse 1 participant
# df <- df |> filter(id %in% c("debugQfD0W:debug1rA1P")) |> glimpse()

# exclude by time
# df <- df |>
#   filter(time >= 1657293679472) |>
#   glimpse()

# remove debug attempts
# df <- df |>
#   filter(!str_detect(id, "debug")) |>
#   glimpse()

# check no duplicate ids
expect_equal(nrow(df |>
                    group_by(id) |>
                    filter(n() > 1)), 0)

# anonymize id
df <- df |>
  mutate(id = fct_anon(factor(id))) |>
  glimpse()

# get nested JSON trial data
df <- df |>
  rowwise() |>
  mutate(trialData = fromJSON(trialData)) |>
  unnest(trialData) |>
  glimpse()

# save data
# df |> write_csv("experiments/chemicals/invariance/wrangled_data.csv")
df |> write_csv("invariance/wrangled_data.csv")

# restrict to main trials
df <- df |>
  filter(trial == "results") |>
  glimpse()

# restrict to practice trials
# df <- df |>
#   filter(trial == "practice") |>
#   glimpse()

# get mean rt
df |>
  group_by(id) |>
  ggplot(aes(mean(rt) / 1000)) + geom_histogram(binwidth = 1)

# exclude participants with mean rt < MIN_AVG_RT
df <- df |>
  group_by(id) |>
  filter(mean(rt) > MIN_AVG_RT) |>
  ungroup()

# Visualize ---------------------------------------------------------------
df <- df |>
  rename(img = imgNum) |>
  select(id, img, responseCompare, robotCondition, labelCompare) |>
  glimpse()

df <- df |>
  mutate(responseCompare = fct_relevel(factor(responseCompare), ref = "same"), robotCondition = fct_relevel(factor(robotCondition), ref = "same"), labelCompare = fct_relevel(factor(labelCompare), ref = "same"),) |>
  glimpse()

df |>
  group_by(robotCondition, labelCompare, responseCompare) |>
  summarise(n())

df |>
  group_by(robotCondition, labelCompare, id, responseCompare) |>
  summarise(n = n()) |>
  group_by(robotCondition, labelCompare) |>
  complete(id, responseCompare, fill = list(n = 0)) |>
  group_by(robotCondition, labelCompare, id) |>
  mutate(p = n / sum(n)) |>
  ggplot(aes(responseCompare, p)) +
  geom_boxplot() +
  geom_jitter() +
  facet_grid(vars(labelCompare), vars(robotCondition), labeller = label_both) +
  labs(y = "P(response)")

# Analyses ----------------------------------------------------------------
df <- df |>
  mutate(choice = ifelse(responseCompare == "same", 0, 1), label = ifelse(labelCompare == "same", 0, 1), robot = fct_relevel(factor(robotCondition), ref = "same"),) |>
  glimpse()

# no need to add interaction with deterministic robots
model_glm <- glm(choice ~ robot + label, data = df, family = binomial)
summary(model_glm)

# use (1 + label | id) since robot between id but labels within id in random robot condition
# add (* | img) gives singular error because no variance
glmer_formula <- as.formula("choice ~ robot + label + (label | id)")
model_glmer <- glmer(glmer_formula, data = df, family = binomial)
summary(model_glmer)

fixef(model_glmer)
ranef(model_glmer)

plot_model(model_glmer, type = "est", transform = NULL, show.intercept = TRUE)
plot_model(model_glmer, type = "re", transform = NULL, show.intercept = TRUE)
plot_model(model_glmer, type = "pred", terms = c("label", "robot"))
plot_model(model_glmer, type = "pred", pred.type = "re", terms = c("label", "id"), show.legend = FALSE, ci.lvl = NA)

# type 3 anova comparing one removed
# mixed(model_glmer,
#       data = df,
#       family = binomial,
#       method = "LRT", # glmer random levels should be > 50
#       test_intercept = TRUE,
#       check_contrasts = FALSE # no interaction means no contrasts coding needed
# )


# Power analysis ----------------------------------------------------------
sqglmer <- safely(.f = quietly(.f = glmer))

es <- 0.362
alpha <- 0.05

get_coefs <- function(id_num, img_num) { sampled_df <- df |>
  nest(data = -c(robotCondition, id)) |>
  group_by(robotCondition) |>
  slice_sample(n = id_num %/% 3, replace = TRUE) |>
  unnest(data) |>
  group_by(id) |>
  slice_sample(n = img_num, replace = TRUE) |>
  ungroup()

  results_list <- sqglmer(glmer_formula, data = sampled_df, family = binomial)

  if (is.null(results_list$error)) {
    # replace model with coefs to save memory
    results_list$result$result <- coef(summary(results_list$result$result)) }
  results_list }

num_reps <- 2

power_df <- expand_grid(id_num = seq(30, 100, length.out = 3), img_num = seq(10, 40, length.out = 3)) |>
  slice(rep(1:n(), times = num_reps)) |>
  glimpse()

plan(multisession, workers = availableCores() - 5)
tic()
multi_power_df <- power_df |>
  mutate(data = future_map2(id_num, img_num, ~get_coefs(.x, .y), .progress = TRUE, .options = furrr_options(seed = TRUE)))
toc()

power_df <- multi_power_df |>
  unnest_wider(data) |>
  filter(is.na(error)) |> # exclude error
  unnest_wider(result) |>
  filter(is.na(warnings) | lengths(warnings) == 0) |> # exclude convergence issues
  glimpse()

power_df <- power_df |>
  rowwise() |>
  mutate(label_sig = (result["label", "Pr(>|z|)"] < alpha & abs(result["label", "Estimate"]) > es), int_sig = (result["(Intercept)", "Pr(>|z|)"] < alpha & abs(result["(Intercept)", "Estimate"]) > es)) |>
  glimpse()

power_df <- power_df |>
  group_by(id_num, img_num) |>
  summarise(label_power = binom.confint(sum(label_sig), num_reps, methods = "wilson"), int_power = binom.confint(sum(int_sig), num_reps, methods = "wilson")) |>
  unnest_wider(c(label_power, int_power), names_sep = "_") |>
  glimpse()

power_df |> ggplot(aes(id_num, img_num)) +
  geom_tile(aes(fill = label_power_mean)) +
  scale_fill_viridis()

power_df |> ggplot(aes(id_num, img_num)) +
  geom_tile(aes(fill = int_power_mean)) +
  scale_fill_viridis()

# Side dishes -------------------------------------------------------------

side <- read_csv("invariance/psiturk/questiondata.csv", col_names = c("id", "question", "response")) |>
  glimpse()

side <- side |>
  mutate(id = fct_anon(factor(id))) |>
  glimpse()

side <- side |> filter(id %in% df$id)

side <- side |>
  pivot_wider(names_from = question, values_from = response) |>
  glimpse()

# df |> write_csv("experiments/chemicals/invariance/wrangled_side.csv")
df |> write_csv("invariance/wrangled_side.csv")

side |>
  group_by(gender) |>
  count()

side |>
  ggplot(aes(as.numeric(age))) + geom_dotplot()

side |>
  filter(!is.na(feedback)) |> glimpse()

side |>
  ggplot(aes(as.numeric(meanMainRT) / 1000)) + geom_dotplot()

side |>
  ggplot(aes(as.numeric(timeElapsed) / 1000 / 60)) + geom_dotplot()
