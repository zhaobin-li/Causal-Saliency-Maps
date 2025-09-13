library(jsonlite)

library(tidyverse)
library(ggplot2)

library(lme4)
library(afex)


# Wrangle ------------------------------------------------------------


# column names in https://psiturk.readthedocs.io/en/stable/command_line.html?highlight=trialdata.csv#download-datafiles
df <-
  read_csv("trialdata_random.csv",
    col_names = c("id", "trialNum", "time", "trialData")
  ) |>
  arrange(time) |> # arrange by time
  glimpse()

# get nested JSON trial data
df <-
  df |>
  rowwise() |>
  mutate(trialData = fromJSON(trialData)) |>
  unnest(trialData) |>
  glimpse()

# restrict to main trials
df <- df |>
  filter(trial == "results") |>
  glimpse()

df <- df |>
  group_by(id) |>
  mutate(id = cur_group_id()) |>
  glimpse()

# Visualize ---------------------------------------------------------------

df <-
  df |>
  select(id, imgNum, responseCompare, robotCompare, labelCompare) |>
  glimpse()

# same == 0, opposite == 1
df <- df |>
  mutate(
    choice = ifelse(responseCompare == "same", 0, 1),
    label = ifelse(labelCompare == "same", 0, 1),
    robot = relevel(factor(robotCompare), ref = "sameRobot")
  ) |>
  glimpse()

df |>
  group_by(robotCompare, labelCompare, responseCompare) |>
  tally()

df |>
  group_by(id, robot, label, choice) |>
  tally() |>
  ggplot(aes(factor(choice), n)) +
  geom_boxplot() +
  geom_jitter() +
  facet_grid(cols = vars(robot, label), labeller = label_context)


# Analyze ----------------------------------------------------------------


model.glm <- glm(choice ~ robot * label,
  data = df,
  family = binomial(link = "logit")
)
summary(model.glm)

# singular error since all participants and images are i.i.d
model.glmer <-
  glmer(choice ~ label * robot + (1 | id) + (1 | imgNum),
    data = df,
    family = binomial
  )

summary(model.glmer)


# Simulate H1 Label Only -------------------------------------------------------------


choiceEqualLabelPr <- 3 / 4

df <- df |>
  rowwise() |>
  mutate(choiceH1LabelOnly = sample(
    c(label, 1 - label),
    1,
    prob = c(
      choiceEqualLabelPr,
      1 - choiceEqualLabelPr
    )
  )) |>
  ungroup()

df |>
  summarize(mean(choiceH1LabelOnly == label))

df |>
  group_by(id, robot, label, choiceH1LabelOnly) |>
  tally() |>
  ggplot(aes(factor(choiceH1LabelOnly), n)) +
  geom_point() +
  geom_boxplot() +
  facet_grid(cols = vars(robot, label), labeller = label_context)

model.H1Label.glm <- glm(choiceH1LabelOnly ~ robot * label,
  data = df,
  family = binomial(link = "logit")
)
summary(model.H1Label.glm)

model.H1Label.glmer <-
  glmer(choiceH1LabelOnly ~ label * robot + (1 | id),
    data = df,
    family = binomial()
  )

summary(model.H1Label.glmer)

model.H1Label.glmer <-
  glmer(choiceH1LabelOnly ~ label + (1 | id) + (1 | imgNum),
    data = df,
    family = binomial()
  )

summary(model.H1Label.glmer)

library(simr)
library(mixedpower)

fixef(model.H1Label.glmer)

power_FLP <- mixedpower(
  model = model.H1Label.glmer, data = df,
  fixed_effects = c("label"),
  simvar = "id", steps = c(5, 15, 20),
  critical_value = 2, n_sim = 10
)

power_FLP

power_SESOI <- mixedpower(
  model = model.H1Label.glmer, data = df,
  fixed_effects = c("label"),
  simvar = "id", steps = c(5, 15, 20),
  critical_value = 2, n_sim = 10, 
  databased = TRUE, SESOI = c(-0.5, 0.5)
)

power_SESOI

multiplotPower(power_SESOI)

R2power <- R2power(model = model.H1Label.glmer, data = df,
                   fixed_effects = c("label"),
                   simvar = "id", steps = c(5, 15, 20),
                   critical_value = 2, n_sim = 10, 
                   databased = TRUE, SESOI = c(-0.5, 0.5),
                   R2var = "imgNum", R2level = 10)

R2power

multiplotPower(R2power)

power <- powerSim(model.H1Label.glmer, nsim = 10, fixed(xname = "label", method = "z"))
power

curve <- powerCurve(model.H1Label.glmer, along = "id", nsim = 10, fixed(xname = "label", method = "z"), breaks = c(4, 6, 8, 10))
print(curve)
summary(curve)
confint(curve)

fixef(model.H1Label.glmer)["label"] <- 0.5
power <- powerSim(model.H1Label.glmer, nsim = 10, fixed(xname = "label", method = "z"))
power

curve <- powerCurve(model.H1Label.glmer, along = "id", nsim = 10, fixed(xname = "label", method = "z"), breaks = c(4, 6, 8, 10))
print(curve)
summary(curve)
confint(curve)

curve$errors

model.2 <- extend(model.H1Label.glmer, along = "id", n = 2)
powerSim(model.2, nsim = 10, fixed(xname = "label", method = "z"))

# Simulate H1 Label + Robot -------------------------------------------------------------


choiceEqualLabelPr <- 3 / 4

df <- df |>
  rowwise() |>
  mutate(choiceH1LabelRobot = case_when(
    robot %in% c("sameRobot", "sameLabel") ~ sample(
      c(label, 1 - label),
      1,
      prob = c(choiceEqualLabelPr, 1 - choiceEqualLabelPr)
    ),
    TRUE ~ choice
  )) |>
  ungroup()

df |>
  group_by(robot) |>
  summarize(mean(choiceH1LabelRobot == label))

df |>
  group_by(id, robot, label, choiceH1LabelRobot) |>
  tally() |>
  ggplot(aes(factor(choiceH1LabelRobot), n)) +
  geom_point() +
  geom_boxplot() +
  facet_grid(cols = vars(robot, label), labeller = label_context)

model.H1LabelRobot <- glm(
  choiceH1LabelRobot ~ robot * label,
  data =  df,
  family = binomial(link = "logit")
)
summary(model.H1LabelRobot)


model <-
  glmer(choiceH1LabelRobot ~ label * robot + (1 | id),
    data = df,
    family = binomial
  )
#
summary(model)
ranef(model)
#
all_fit(model)
#
