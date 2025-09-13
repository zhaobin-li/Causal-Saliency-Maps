Experiment setup [guide](https://psiturk.readthedocs.io/en/stable/quickstart.html)

# Running now

## Python environment

```bash
pip install -r experiments/chemicals/requirements.txt
```

## Counterfactual Experiment:

### Directory: `experiments/chemicals/counterfactual/psiturk`

### App: [https://xai-chemical-app.herokuapp.com](https://xai-chemical-app.herokuapp.com)

## Implementation Invariance Experiment:

### Directory: `experiments/chemicals/invariance/psiturk`

### App: [https://xai-robot-app.herokuapp.com](https://xai-robot-app.herokuapp.com)

# Basic setup

## Installation command

The command below will install everything python, though you may need to install additional non-python libraries based
on the instructions below.

```bash
pip install psiturk psycopg2-binary numpy matplotlib seaborn networkx tqdm pygraphviz
```

## PsiTurk Example

Latest version (3.3.0) supports [python 3.8.10](https://github.com/NYUCCL/psiTurk/actions/runs/1912409852) (latest 3.8
version with prebuilt binary on macOS). Install and setup PsiTurk Stroop example:

```bash
pip install psiturk psycopg2-binary # psycopg2-binary connects to Heroku postgres database
psiturk-setup-example
```

All terminal commands below will run in psiturk-setup-example

## Graphing molecules

To graph the molecules, install

```bash
pip install numpy matplotlib seaborn networkx tqdm
```

and [pygraphviz](https://pygraphviz.github.io/documentation/stable/install.html) using (latest version 3.0.0 on MacOS)

```bash
brew install graphviz 
pip install pygraphviz
```

Then run `molecules.py` and get colorbar using `colorbar.py`. The images will be saved in `psiturk/static/stimuli`.

Additional packages may be required on [MacOS](#Additional-packages). I have also pinned the local production
environment using `pip freeze > experiments/chemicals/requirements.txt` and seeded `molecules.py` to ensure
repeatability. You can install the required packages (excluding additional packages and graphviz) using

```bash
pip install -r experiments/chemicals/requirements.txt
```

Note: The `requirements.txt` in the two `psiturk` directories are to be deployed online by Heroku

## MTurker Requester Sandbox [Account](https://docs.aws.amazon.com/AWSMechTurk/latest/AWSMechanicalTurkGettingStartedGuide/SetUp.html#setup-aws-account)

1. Remember to get a sandbox account
2. Better to use IAM credentials

and boto3 (installed with psiturk) by storing IAM credentials in `~/.aws/credentials`

```text
[default]
aws_access_key_id = YOUR_ACCESS_KEY
aws_secret_access_key = YOUR_SECRET_KEY
```

and optionally default region as us-east-1 (also set by psiturk) in `~/.aws/config`

```text
[default]
region=us-east-1
```

Check MTurk
setup [using](https://docs.aws.amazon.com/AWSMechTurk/latest/AWSMechanicalTurkRequester/mturk-use-sandbox.html)

```python
import boto3

client = boto3.client(
    'mturk',
    endpoint_url='https://mturk-requester-sandbox.us-east-1.amazonaws.com'  # sandbox
    # endpoint_url='https://mturk-requester.us-east-1.amazonaws.com'  # production
)
print(client.get_account_balance()['AvailableBalance'])  # 10000.00 in sandbox
```

## Heroku [App](https://www.heroku.com/)

Create an account, install [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli) and check with

```bash
heroku --version
```

Login to Heroku using

```bash
heroku login
```

A browser will pop up and the authentication is saved at `~/.netrc`. Consider upgrading to app
to [hobbyist](https://devcenter.heroku.com/articles/dyno-types) to prevent app sleep.

# Run experiment

## Start local PsiTurk

Only to debug locally, as once uploaded to Heroku and connected to MTurk, PsiTurk server will always run on Heroku and
accessible through MTurk.

```bash
cd psiturk-example
psiturk
server on
```

to start the psiTurk shell. Run

```psiTurk shell
debug
```

to launch the experiment with a unique debug url (whether on local machine or Heroku depending on `ad_url_domain`
in `config.txt`)

## Initialize Heroku app

```bash
git init
heroku create [optional: app name] # or heroku git:remote -a [app name] to attach to existing app 
psiturk-heroku-config
```

and setup database using

```bash
heroku addons:create heroku-postgresql
```

Check app url with `heroku domains`, database url with `heroku config`, addons with `heroku addons` and more
with `heroku pg` anytime. **Anyone can connect to the database using the `DATABASE_URL`.**

## PsiTurk on Heroku

I have set the python runtime version
to be [3.8.13 as supported on Heroku](https://devcenter.heroku.com/articles/python-support#supported-runtimes)
in `runtime.txt` and the package versions in `requirements.txt`.

To upload using git:

```bash
git add .
git commit -m "Initial commit"
git push heroku master
```

Check the app on the cloud using:

```bash
heroku open
```

Connect with Heroku PostgreSQL server (get using `heroku config`) by setting `database_url` in `config.txt` (will expose
database in experiment code). Download database to `trialdata.csv`, `questiondata.csv`, and `eventdata.csv`

```bash
psiturk download_datafiles
```

## Heroku on MTurk

Point Heroku to MTurk by setting `ad_url_domain` in `config.txt` to your Heroku app (
e.g. `ad_url_domain = example-app.herokuapp.com`). Set `title` (e.g. `title = Stroop task`) and `description` (
e.g. `description = Judge the color of a series of words.`).

Launch psiTurk shell and run

```psiTurk shell
hit create
```

to launch in sandbox.

Check HITs and workers using

```psiTurk shell
hit list 
worker list
```

HIT status should be assignable now and max more than pending + remain.

Approve workers and expire HIT using

```psiTurk shell
worker approve --all
hit expire --all
```

HIT status says `Reviewable` and max equals complete when you are done reviewing workers

### Live

Reset the database using

```bash
heroku pg:reset DATABASE
```

Set `approve_requirement = 95` and `number_hits_approved = 100` in `config.txt`, restart psiTurk and begin live using

```psiTurk shell
mode
```

and repeat the steps in the [Heroku on MTurk](Heroku-on-MTurk) section.

# jsPsych in psiTurk

I will use [Hello World](https://www.jspsych.org/7.2/tutorials/hello-world/) as a demo.

## HTML

psiTurk uses Jinja2's template engine to render html in the `templates` directory.
To use jsPsych in psiTurk, add jsPsych.js, the plugins and css to `templates/exp.html`
within `{% block head %}{% endblock %}` _before_ loading `/static/js/task.js`.

```html
{% block head %}

<script src="https://unpkg.com/jspsych@7.2.3" type="text/javascript"></script>
<!--add your plugins-->
<script src="https://unpkg.com/@jspsych/plugin-html-keyboard-response@1.1.1"></script>
<!--add your plugins-->
<link href="https://unpkg.com/jspsych@7.2.3/css/jspsych.css" rel="stylesheet" type="text/css"/>

<!-- task.js is where you experiment code actually lives
    for most purposes this is where you want to focus debugging, development, etc...
-->
<script src="/static/js/task.js" type="text/javascript"></script>

{% endblock %}
```

## JS

Then clear `static/js/task.js` and add jsPsych code. At the beginning, include the line `... new PsiTurk(...);` to
communicate with the psiTurk server and add some psiTurk callbacks to `initJsPsych(...)` to save data to psiTurk
database. Upon calling `psiturk download_datafiles`, those data saved with `psiturk.recordTrialData()` will
be [stored](https://psiturk.readthedocs.io/en/stable/command_line.html#download-datafiles) in `trialdata.csv`.

For instance using CDN-based setup:

```javascript
const psiTurk = new PsiTurk(uniqueId, adServerLoc, mode);

// between-subjects condition and within-subjects counterbalance variables
console.log(condition);
console.log(counterbalance);

const jsPsych = initJsPsych({
    on_finish: function (data) {
        psiTurk.recordTrialData(data);
        psiTurk.saveData({
            success: () => psiTurk.completeHIT(),
        });
    },
});

const hello_trial = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: 'Hello world!'
}

jsPsych.run([hello_trial]);
```

Then change the reward and time in `templates/ad.html`, [switch on](#Start-local-PsiTurk) the psiTurk server and run the
experiment.

## Actual experiment

I have downloaded jspsych 7.2.3 and added `static/css/chemicals.css` and `static/js/chemicals.js` as helpers
to `templates/exp.html` etc. Allowing Chrome only in jsPsych since the experiment is only tested on Chrome -- Firefox
has issues loading the molecules `.svg` embedded with raster saliency map.

# Analysis

In both `counterfactual` and `invariance` directories, the data is mainly contained
in `trialdata.csv` and some additional statistics
in `questiondata.csv` (https://psiturk.readthedocs.io/en/stable/command_line.html?highlight=trialdata.csv#download-datafiles)
. The analysis script is `analysis.R` in both directories (in RStudio open `experiments/chemicals/chemicals.Rproj`)

# Configuration

## Precedence

Lowest priority 1st:

1. Defaults in `config.txt` (commented with `;`)
2. Defaults in Heroku
3. Settings in `config.txt`
4. Settings in Heroku named `PSITURK_{uppercase_config_name}`
   (using `heroku config:set`, e.g. `heroku config:set PSITURK_THREADS=1`)
   except `DATABASE_URL` and `PORT` which always take precedence

## Config.txt

[Check documentation](https://psiturk.readthedocs.io/en/stable/settings.html?highlight=config.txt). Below are some
pointers.

### HIT Configuration

Set the HIT attributes and eligibility. I would not recommend requiring [pricier](https://www.mturk.com/pricing) master
workers, since determining their status
is [a black box](https://www.mturk.com/worker/help#what_is_master_worker). Remember to set
the [ad_url_domain](#Heroku-on-MTurk)

### Database Parameters

Set [database](#PsiTurk-on-Heroku). Besides downloading the data using `psiturk download_datafiles`, you can connect to
the database directly and run SQL commands.

### Task Parameters

Set contact email etc. Upon setting between and within-participants condition, remember to update
the `experiment_code_version` and/or reset database with every deployment to reset the condition randomization!

# Debugging

## Additional packages

On MacOS, I had to install

```bash
pip install gnureadline # psiturk shell
```

## Debugging Heroku

Check the activity page and build log in Heroku app. Running `psiturk-heroku-config` copies these into `psiturk-example`
to compile on Heroku upon git upload. One may have to change them as needed.

1. Procfile
2. requirements.txt
3. herokuapp.py

# Bonus

To compute bonus, replace `psiTurk.saveData(...)` in `static/js/task.js` with

```javascript
 psiTurk.saveData({
    success: () =>
        psiTurk.computeBonus("compute_bonus", () => {
            psiTurk.completeHIT(); // when finished saving compute bonus, the quit
        }),
});
```

and change `compute_bonus` in `custom.py`. For instance, I calculated the proportion correct of attention trials
in `static/js/task.js` and recorded it using `psiTurk.recordUnstructuredData("proportionCorrect", proportionCorrect);`.
Then to award $0.50 to workers whose proportion correct is greater than 0.8:

```python
if user_data['questiondata']["proportionCorrect"] >= 0.8:
    bonus += 0.50
```

To actually grant the bonus, call both lines

```psiturk
worker bonus --auto --all --reason="Thanks for paying attention!"
```