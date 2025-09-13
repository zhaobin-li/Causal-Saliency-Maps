#' ---
#' title: " Experiment Counterfactual Power Analysis"
#' author: "ZhaoBin Li, Scott Cheng-Hsin Yang, Tomas Folke, Patrick Shafto"
#' date: "Aug 3rd, 2022"
#' ---
#'
# conda create -n r_chemicals -c conda-forge r-base r-jsonlite r-tidyverse r-ggplot2 r-lme4 r-afex r-furrr r-tictoc -y
library(viridis)
library(binom)

library(tictoc)
library(furrr)

library(sjPlot)
library(afex)
library(lme4)

library(jsonlite)
library(testthat)

library(ggplot2)
library(tidyverse)

MIN_PROPROTION_CORRECT <- 0.8 # attention check threshold

# Wrangle ------------------------------------------------------------

# column names in https://psiturk.readthedocs.io/en/stable/command_line.html?highlight=trialdata.csv#download-datafiles
df <-
  # read_csv("experiments/chemicals/counterfactual/psiturk/trialdata.csv",
  read_csv("counterfactual/psiturk/trialdata.csv", col_names = c("id", "trialNum", "time", "trialData")) |>
  arrange(time) |> # arrange by time
  glimpse()

# analyse 1 participant
# df <- df |> filter(id %in% c("debugaRXOD:debugRJE75")) |> glimpse()

# exclude by time
# df <- df |>
#   filter(time >= 1662435024026) |>
#   glimpse()
# 
# remove debug attempts
df <- df |>
  filter(!str_detect(id, "debug")) |>
  glimpse()

# exclude Pat's and ZB's ids
df <- df |>
  filter(!str_detect(id, "ABYGLITD9LAX5|A2ISY01HC7N0RA")) |>
  glimpse()

# check no duplicate ids
expect_equal(nrow(df |>
  group_by(id) |>
  filter(n() > 1)), 0)

# save id
included_ids <- df |> pull(id)

# Side dishes -------------------------------------------------------------

side <- read_csv("counterfactual/psiturk/questiondata.csv", col_names = c("id", "question", "response")) |>
  glimpse()

side <- side |>
  filter(id %in% included_ids) |>
  glimpse()

side <- side |>
  pivot_wider(names_from = question, values_from = response) |>
  glimpse()

# check no duplicate ids
expect_equal(nrow(side |>
  group_by(id) |>
  filter(n() > 1)), 0)

side <- side |> mutate(
  proportionCorrect = as.numeric(proportionCorrect),
  age = round(as.numeric(age)),
  meanMainRTsecs = round(as.numeric(meanMainRTsecs)),
  totalTimeElapsedmins = round(as.numeric(totalTimeElapsedmins))
) |> glimpse()

side |>
  group_by(gender) |>
  count()

side |>
  ggplot(aes(age)) +
  geom_dotplot()

side |>
  filter(!is.na(feedback))

side |>
  ggplot(aes(meanMainRTsecs)) +
  geom_dotplot()

side |>
  ggplot(aes(totalTimeElapsedmins)) +
  geom_dotplot()

side |>
  ggplot(aes(proportionCorrect)) +
  geom_dotplot()

side |> summarize(sum(proportionCorrect >= MIN_PROPROTION_CORRECT))

excluded_ids <- side |>
  filter(proportionCorrect < MIN_PROPROTION_CORRECT) |>
  pull(id)

side <- side |>
  mutate(id = fct_anon(factor(id))) |>
  glimpse()

# side |> write_csv("experiments/chemicals/counterfactual/wrangled_side.csv")
side |> write_csv("counterfactual/wrangled_side.csv")

# Checks and exclusion ---------------------------------------------------------------

df <- df |>
  filter(!(id %in% excluded_ids)) |>
  glimpse()

# anonymize id
df <- df |>
  mutate(id = fct_anon(factor(id))) |>
  glimpse()

# save data
# df |> write_csv("experiments/chemicals/counterfactual/wrangled_data.csv")
df |> write_csv("counterfactual/wrangled_data.csv")

# get nested JSON trial data
df <- df |>
  rowwise() |>
  mutate(trialData = fromJSON(trialData)) |>
  unnest(trialData) |>
  glimpse()

# restrict to main trials
df <- df |>
  filter(trial == "results") |>
  glimpse()

# restrict to practice trials
# df <- df |>
#   filter(trial == "practice") |>
#   glimpse()

# discard attention trials
df <- df |>
  filter(isAttention == FALSE) |>
  glimpse()

# Visualize ---------------------------------------------------------------

df <- df |>
  rename(img = imgNum) |>
  select(id, img, labelCompare, promptImg) |>
  glimpse()

df <- df |>
  mutate(
    labelCompare = fct_relevel(factor(labelCompare), ref = "same"),
    promptImg = fct_relevel(factor(promptImg), ref = "same")
  ) |>
  glimpse()

df |>
  group_by(promptImg, labelCompare) |>
  summarise(n())

df |>
  group_by(promptImg, id, labelCompare) |>
  summarize(n = n()) |>
  group_by(promptImg) |>
  complete(id, labelCompare, fill = list(n = 0)) |>
  group_by(promptImg, id) |>
  mutate(p = n / sum(n)) |>
  filter(labelCompare == "opp") |>
  ggplot(aes(promptImg, p)) +
  geom_boxplot() +
  ylim(0, 1) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "red", size = 2) +
  labs(y = 'P(labelCompare == "opp")')

df |>
  group_by(promptImg, id, labelCompare) |>
  summarize(n = n()) |>
  group_by(promptImg) |>
  complete(id, labelCompare, fill = list(n = 0)) |>
  group_by(promptImg, id) |>
  mutate(p = n / sum(n)) |>
  filter(labelCompare == "opp") |>
  ggplot(aes(promptImg, p)) +
  geom_line(aes(group = id, color = id)) +
  ylim(0, 1) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "red", size = 2) +
  labs(y = 'P(labelCompare == "opp")')

# Analyze ----------------------------------------------------------------
df <- df |>
  mutate(
    choice = ifelse(labelCompare == "same", 0, 1),
    label = ifelse(promptImg == "same", 0, 1),
  ) |>
  glimpse()

model_glm <- glm(choice ~ label, data = df, family = binomial)
summary(model_glm)

# intercept model
glmer_formula <- as.formula("choice ~ label + (1 | id) + (1 | img)")
model_glmer <- glmer(glmer_formula, data = df, family = binomial)
summary(model_glmer)

# maximal model
glmer_formula <- as.formula("choice ~ label + (label | id) + (label | img)")
model_glmer <- glmer(glmer_formula, data = df, family = binomial)
summary(model_glmer)

# intercept participant model
glmer_formula <- as.formula("choice ~ label + (1 | id)")
model_glmer <- glmer(glmer_formula, data = df, family = binomial)
summary(model_glmer)

# (1 + label | id) since labels within id
# add (* | img) gives singular error because no variance
glmer_formula <- as.formula("choice ~ label + (label | id)")
model_glmer <- glmer(glmer_formula, data = df, family = binomial)
summary(model_glmer)

fixef(model_glmer)
ranef(model_glmer)

plot_model(model_glmer, type = "est", transform = NULL, show.intercept = TRUE)
plot_model(model_glmer, type = "re", transform = NULL, show.intercept = TRUE)
plot_model(model_glmer, type = "pred", terms = "label")
plot_model(model_glmer, type = "pred", pred.type = "re", terms = c("label", "id"), show.legend = FALSE, ci.lvl = NA)

# type 3 anova comparing one removed
# mixed(model_glmer,
#       data = df,
#       family = binomial,
#       method = "LRT", # glmer random levels should be > 50
#       test_intercept = TRUE,
#       check_contrasts = FALSE # no interaction means no contrasts coding needed
# )

df <- df |>
  mutate(
    choice = ifelse(labelCompare == "same", 1, 0),
    label = ifelse(promptImg == "same", 1, 0),
  ) |>
  glimpse()

model_glm <- glm(choice ~ label, data = df, family = binomial)
summary(model_glm)

# (1 + label | id) since labels within id
# add (* | img) gives singular error because no variance
glmer_formula <- as.formula("choice ~ label + (label | id)")
model_glmer <- glmer(glmer_formula, data = df, family = binomial)
summary(model_glmer)

fixef(model_glmer)
ranef(model_glmer)

plot_model(model_glmer, type = "est", transform = NULL, show.intercept = TRUE)
plot_model(model_glmer, type = "re", transform = NULL, show.intercept = TRUE)
plot_model(model_glmer, type = "pred", terms = "label")
plot_model(model_glmer, type = "pred", pred.type = "re", terms = c("label", "id"), show.legend = FALSE, ci.lvl = NA)


# Power analysis ----------------------------------------------------------
sqglmer <- safely(.f = quietly(.f = glmer))

es <- 0.362
alpha <- 0.05

get_coefs <- function(id_num, img_num) {
  sampled_df <- df |>
    nest(data = -id) |>
    slice_sample(n = id_num, replace = TRUE) |>
    unnest(data) |>
    group_by(id) |>
    slice_sample(n = img_num, replace = TRUE) |>
    ungroup()

  results_list <- sqglmer(glmer_formula, data = sampled_df, family = binomial)

  if (is.null(results_list$error)) {
    # replace model with coefs to save memory
    results_list$result$result <- coef(summary(results_list$result$result))
  }
  results_list
}

num_reps <- 2

power_df <- expand_grid(id_num = seq(30, 100, length.out = 3), img_num = seq(10, 40, length.out = 3)) |>
  slice(rep(1:n(), times = num_reps)) |>
  glimpse()

plan(multisession, workers = availableCores() - 5)
tic()
multi_power_df <- power_df |>
  mutate(data = future_map2(id_num, img_num, ~ get_coefs(.x, .y), .progress = TRUE, .options = furrr_options(seed = TRUE)))
toc()

power_df <- multi_power_df |>
  unnest_wider(data) |>
  filter(is.na(error)) |> # exclude error
  unnest_wider(result) |>
  filter(is.na(warnings) | lengths(warnings) == 0) |> # exclude convergence issues
  glimpse()

power_df <- power_df |>
  rowwise() |>
  mutate(
    label_sig = (result["label", "Pr(>|z|)"] < alpha &
      result["label", "Estimate"] > es &
      (result["label", "Estimate"] + result["(Intercept)", "Estimate"]) > es),
    int_sig = (result["(Intercept)", "Pr(>|z|)"] < alpha &
      result["(Intercept)", "Estimate"] < -es)
  ) |>
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
