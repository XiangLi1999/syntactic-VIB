#!/usr/bin/python
# --------------------------------------- 
# File Name : shepherd3.py
# Creation Date : 21-11-2017
# Last Modified : Tue Nov 21 11:13:40 2017
# Created By : wdd 
# --------------------------------------- 
import argparse, re, os, math, sys
import numpy.random as rnd
from functools import wraps
import random
import json
import logging
import inspect
import subprocess

_aparser = argparse.ArgumentParser()
_aparser.add_argument('task', default='main', type=str, action='store', help='')
_aparser.add_argument('--prfx', type=str, default='')
_aparser.add_argument('--workspace', type=str, default='')
_aparser.add_argument('--data', type=str, default='')
_aparser.add_argument('--out', type=str, default='')
_aparser.add_argument('--log', type=str, default='')
_aparser.add_argument('--skip', type=str, nargs='+', default='')
_aparser.add_argument('--local', action='store_true', help='Dry run')
_aparser.add_argument('--overwrite', action='store_true', help='Dry run')
_aparser.add_argument('-s', '--sys', type=str, nargs='+', default='')
_aparser.add_argument('-u', '--usr', type=str, nargs='+', default='')
_aparser.add_argument('-d', '--dry', action='store_true', help='Dry run')
_aparser.add_argument('-e', '--est', action='store_true', help='Dry run')
_aparser.add_argument('-l', '--latest', action='store_true', help='Dry run')
_aparser.add_argument('-b', '--bs', type=int, default=1, help='Batch size')
_aparser.add_argument('-r', '--sr', type=float, default=1., help='Grid search rate')
_args = _aparser.parse_args()


class VarDict(object):
    @staticmethod
    def _setattr_(obj, key, val):
        obj.my_dict[key] = val

    @staticmethod
    def _getattr_(obj, key):
        return obj.my_dict[key]

    def __init__(self, dict=None):
        self.__dict__['my_dict'] = {}
        if dict:
            for key, val in dict.items():
                self.__setattr__(key, val)

    def __setitem__(self, key, value):
        VarDict._setattr_(self, key, value)

    def __getitem__(self, key):
        return VarDict._getattr_(self, key)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __getattr__(self, key):
        return self.__getitem__(key)

    def __contains__(self, key):
        return key in self.my_dict

    def __str__(self):
        return '\n'.join('{0}:{1}'.format(key, val) for (key, val) in sorted(self.my_dict.items()))

    def get(self, key, value):
        return self.__getitem__(key) if key in self.my_dict else value

    def set(self, key, value, func=lambda x: x):
        val = func(self.get(key, value))
        self.__setattr__(key, val)
        return val

    def to_dict(self):
        return self.my_dict

    def add(self, dict):
        for key, val in dict.items():
            self.__setattr__(key, val)


def _collect_exp_functions(fn):
    state, functions = 0, set()
    p = re.compile('def\s+(.*?)\(')
    with open(fn, 'r') as f:
        for l in f:
            if l.startswith('@shepherd('):
                state = 1
            elif state:
                if l.startswith('@') or l.startswith('#'):
                    continue
                m = p.match(l)
                if m:
                    functions.add(m.group(1))
                state = 0
    return functions


CONF = VarDict()
SYS = VarDict()
USR = VarDict()


def ALL():
    ret = dict(map(lambda x: ('S_' + x[0], x[1]), SYS.to_dict().items()))
    ret.update(dict(map(lambda x: ('U_' + x[0], x[1]), USR.to_dict().items())))
    return ret


def get_logger(level=logging.INFO):
    logger = logging.getLogger(os.path.basename(inspect.getouterframes(inspect.currentframe())[1][1]))
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s-%(name)s[%(levelname)s]$ %(message)s', '%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(SYS.syslog)
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


logger = None

_handler = None
EXP_FILE = 'experiment.py'
MAXINT = 655353
_line_splitter = re.compile('\s+')


def SPD():
    return _handler


def CMD(x, log=True):
    if log:
        logger.info('<BASH>:%s' % x)
    return subprocess.check_output(x, shell=True).decode('utf-8').strip()


def CMD_LOACL(x, log=True):
    if log:
        logger.info('<BASH>:%s' % x)
    p = subprocess.Popen(x, shell=True, stdout=subprocess.PIPE)
    while p.poll() is None:
        l = p.stdout.readline().decode('utf8')
        print(l.strip())
    print(p.stdout.read().decode('utf8').strip())


def DRY_RUN(command):
    logger.info('dry:%s' % command)
    return 'dry'


def _load_conf(conf='.spdrc.json', var_dict=SYS):
    if os.path.isfile(conf):
        with open(conf) as json_data:
            var_dict.add(json.load(json_data))


def _list_to_dict(alist):
    def _to_named_args(itm):
        if '=' in itm:
            idx = itm.find('=')
            return (itm[:idx], itm[idx + 1:])
        else:
            return (itm, '1')

    return dict(map(_to_named_args, alist))


def shepherd(before=[], after=[]):
    def decorator(func):
        @wraps(func)
        def decorated(*args, **kwargs):
            for bfr in before: bfr()
            try:
                func(*args, **kwargs)
            except KeyboardInterrupt:
                pass
            for afr in after: afr()

        return decorated

    return decorator


def make_dirs():
    SYS.jobs = '%(out)s/jobs' % SYS
    SYS.task_dir = '%(jobs)s/%(task_name)s' % SYS
    SYS.std = '%(task_dir)s/std' % SYS
    SYS.tmp = '%(task_dir)s/tmp' % SYS
    SYS.log = '%(task_dir)s/log' % SYS
    SYS.model = '%(task_dir)s/model' % SYS
    SYS.src = '%(task_dir)s/src' % SYS
    SYS.scripts = '%(task_dir)s/scripts' % SYS
    SYS.output = '%(task_dir)s/output' % SYS
    SYS.evaluation = '%(task_dir)s/eval' % SYS
    SYS.test_evaluation = '%(evaluation)s/test' % SYS
    for dir in [SYS.jobs,
                SYS.task_dir,
                SYS.std, SYS.tmp, SYS.log,
                SYS.model,
                SYS.src,
                SYS.scripts,
                SYS.output,
                SYS.evaluation,
                SYS.test_evaluation]:
        if not os.path.exists(dir):
            os.makedirs(dir)

def setup():
    _load_conf()
    SYS.add(_list_to_dict(_args.sys))
    USR.add(_list_to_dict(_args.usr))
    SYS.set('task_prfx', '')
    SYS.set('workspace', os.getcwd())
    SYS.set('data', os.path.join('%(workspace)s' % SYS, 'data'))
    SYS.set('out', os.getcwd())

    log_dir = os.path.join('%(workspace)s' % SYS, 'log')
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    SYS['syslog'] = os.path.join(log_dir, SYS.set('syslog', 'log.txt'))
    if _args.prfx: SYS.task_prfx = _args.prfx
    if _args.workspace: SYS.workspace = _args.workspace
    if _args.out: SYS.out = _args.out
    if _args.log: SYS.syslog = _args.log
    global logger
    logger = get_logger()
    logger.info('<EXPERIMENT>:%s' % ' '.join(sys.argv))


def load_job_info(info_file, key='ctask_list'):
    ret = []
    with open(info_file, 'r') as f:
        f.readline()
        for l in f:
            ret += list(json.loads(l.strip())[key])
    return ret


def init():
    setup()
    task_l = _args.task.split('-')
    if len(task_l) == 1: task_l += ['TEST']
    prfx = '%(task_prfx)s-' % SYS if SYS.task_prfx else ''
    SYS.set('task_name', prfx + '-'.join(task_l))
    SYS.set('device', 'cpu')
    SYS.set('python_itrptr', 'python')
    SYS.set('mem', '5g')
    SYS.time = CMD('date +%Y_%m_%d_%H_%M_%S')
    rnd.seed(SYS.set('seed', 0, int))
    make_dirs()
    SYS.set('skip', set())
    if not _args.overwrite:
        for info in os.listdir(SYS.task_dir):
            if info.startswith('info-'):
                SYS.skip.update(load_job_info(os.path.join(SYS.task_dir, info)))
    for sfn in _args.skip:
        SYS.skip.update(load_job_info(sfn))
    if _args.latest:
        logger.info('Using the local code!')
    try:
        CMD('cp -r -n %(workspace)s/src/* %(src)s' % SYS)
    except:
        if not _args.latest:
            logger.info('Using the old code!')
    if SYS.host == 'local' or _args.local:
        init_local()
    else:
        init_cluster()
    SYS.python_dir = '%(workspace)s/src' % SYS if _args.latest else '%(src)s' % SYS
    logger.info('task name: %(task_name)s' % SYS)


def post():
    _handler.finish()


def init_local():
    global _handler
    _handler = LocalJobHandler()


def init_clsp_grid():
    global _handler
    _handler = CLSPJobHandler()


def init_marcc():
    global _handler
    _handler = MarccInteractJobHandler() \
        if SYS.set('interact', 0, int) else MarccJobHandler()


def init_cluster():
    SYS.set('job_number', 100000)
    SYS.set('wait', '24:00:00')
    if SYS.host == 'clsp':
        init_clsp_grid()
    elif SYS.host == 'marcc':
        init_marcc()


def grid_search(func, search_list, seed=123, rate=_args.sr):
    gs(func, list(map(lambda x: (x[0], USR.get(x[0], x[1]).split('|')), search_list)), None, random.Random(seed), rate)


def gs(func, remained, searched, rand, rate):
    if searched is None:
        if _args.est:
            total = 1
            for _, lst in remained:
                total *= len(lst)
            logger.info('Total Jobs:%i' % total)
            return
        gs(func, remained, [], rand, rate)
    elif len(remained) == 0:
        if rand.random() < rate:
            _handler.submit(*func(searched))
    else:
        name, candidates = remained[0]
        for val in candidates:
            gs(func, remained[1:], searched + [(name, val)], rand, rate)


def arg_conf(val_map):
    return ' '.join('--{0} {1}'.format(k, v) for k, v in val_map), '-'.join(
        '{0}={1}'.format(k[0], v) for k, v in val_map)


def basic_func(header, val_map,
               param_func=lambda x: 'job_' + x,
               jobid_func=lambda x: 'job_' + x):
    args, config = arg_conf(val_map)
    command = header.format(config=param_func(config)) + ' ' + args
    return [command], jobid_func(config)


class JobHandler(object):
    def __init__(self):
        self._job_counter = 0
        self._local_rtask_queue = []
        self._local_ctask_queue = []
        self._global_ctask_queue = []
        self._global_rtask_queue = []
        self._bash_prfx = ''

    def _valid(self, command, config):
        return True

    def _before(self, id):
        self.script = SYS.scripts + '/' + id + '.sh'
        with open(self.script, 'w') as f:
            print(self._bash_prfx.format(id=id), file=f)
            print(SYS.get('before_exe', ''), file=f)
            for cmd_list, config in self._local_rtask_queue:
                for cmd in cmd_list:
                    if type(cmd) == tuple and not cmd[1]:
                        print(cmd[0], file=f)
                    elif type(cmd) == tuple and cmd[1]:
                        print(cmd[0] + ' | tee -a %(std)s/{config}.log'.format(config=config) % SYS, file=f)
                    else:
                        print(cmd + ' | tee -a %(std)s/{config}.log'.format(config=config) % SYS, file=f)
            print(SYS.get('after_exe', ''), file=f)

    def _after(self, id):
        pass

    def _run(self, id):
        pass

    def _submit_queue(self, id):
        self._before(id)
        self._run(id)
        self._after(id)
        self._local_ctask_queue = []
        self._local_rtask_queue = []
        self._job_counter += 1

    def _finish(self):
        pass

    def submit(self, command, config='TEST'):
        self._local_ctask_queue += [config]
        self._global_ctask_queue += [config]
        if not self._valid(command, config):
            logger.info('<SKIP>: ' + config)
            return
        self._local_rtask_queue += [(command, config)]
        self._global_rtask_queue += [config]
        if len(self._local_rtask_queue) == _args.bs:
            self._submit_queue(config)

    def finish(self):
        if len(self._local_rtask_queue) > 0:
            self._submit_queue(self._local_rtask_queue[-1][1])
        self._finish()
        logger.info('%i tasks submitted!' % len(self._global_rtask_queue))
        logger.info('%i jobs submitted!' % self._job_counter)


class LocalJobHandler(JobHandler):
    def __init__(self):
        super(LocalJobHandler, self).__init__()
        self._bash_prfx = '#!/bin/bash -l\n' \
                          '#--job-name=%(task_name)s:{id}\n' % SYS
        self._my_run = DRY_RUN if _args.dry else CMD_LOACL

    def _run(self, id):
        self._my_run('bash {script}'.format(script=self.script))


class CLSPJobHandler(JobHandler):
    def __init__(self):
        super(CLSPJobHandler, self).__init__()
        SYS.set('gpus', '0')
        dev_flag = '#$ -pe smp %(cpus)s\n' % SYS
        if int(SYS.gpus) > 0:
            dev_flag += '#$ -l gpu=%(gpus)s,h=b1[1-8]*\n' \
                        '#$ -q g.q\n' % SYS
            SYS.device = 'gpu'
        self._bash_prfx = \
            '#!/bin/bash\n' \
            '#$ -N %(task_name)s:{id}\n' \
            '#$ -cwd\n' \
            '#$ -l mem_free=%(mem)s,ram_free=%(mem)s,arch=*64*\n' \
            '#$ -o %(std)s/{id}.o\n' \
            '#$ -e %(std)s/{id}.e\n' % SYS + dev_flag

        def _run_grid(message):
            submit_message = CMD(message)
            lines = submit_message.split('\n')
            submit_message = lines[len(lines) - 1]
            return _line_splitter.split(submit_message)[2]

        self._my_run = DRY_RUN if _args.dry else _run_grid

    def _valid(self, command, id):
        if id in SYS.skip:
            return False
        ret = CMD("qstat -u $(whoami) -r | grep 'Full jobname'")
        jid = '%(task_name)s:{id}'.format(id=id) % SYS
        return not len(ret) or jid not in [line.split()[2] for line in ret.split('\n')]

    def _run(self, id):
        self.job_id = self._my_run('qsub {script}'.format(script=self.script))

    def _after(self, id):
        with open(SYS.task_dir + '/.tmp', "a") as f:
            print("qdel {job_id} #{id}".format(job_id=self.job_id, id=id), file=f)

    def _finish(self):
        tmpfile = SYS.task_dir + '/.tmp'
        if os.path.isfile(tmpfile):
            os.rename(tmpfile, SYS.task_dir + '/kill_' + SYS.time + '.sh')


class MarccJobHandler(JobHandler):
    def __init__(self):
        super(MarccJobHandler, self).__init__()
        SYS.set('queue', 'shared')
        mem = float(re.compile('(\d+(\.\d+)?)g').match(SYS.mem).group(1))
        memory_per_cpu = 21.1 if 'mem' in SYS.queue else 5.3
        default_ncpus = int(math.ceil(mem / memory_per_cpu))
        if 'gpu' in SYS.queue:
            cpper_gpu = 6
            default_ngpus = int(math.ceil(mem / (memory_per_cpu * cpper_gpu)))
            SYS.set('gpus', default_ngpus, int)
            SYS.set('cpus', SYS.gpus * cpper_gpu)
            SYS.device = 'gpu'
            dev_flag = '#SBATCH --gres=gpu:%(gpus)s\n' \
                       '#SBATCH --cpus-per-task=%(cpus)s\n' % SYS
        else:
            SYS.set('cpus', default_ncpus)
            dev_flag = '#SBATCH --cpus-per-task=%(cpus)s\n' % SYS
        if 'scav' in SYS.queue:
            SYS.set('requeue', 1, int)
            dev_flag += '#SBATCH --qos=scavenger\n' % SYS
            SYS.set('duration', '6:0:0')
        else:
            SYS.set('duration', '12:0:0')
        self._bash_prfx = \
            '#!/bin/bash\n' \
            '#SBATCH --job-name=%(task_name)s:{id}\n' \
            '#SBATCH --mem=%(mem)s\n' \
            '#SBATCH --ntasks-per-node=1\n' \
            '#SBATCH --time=%(duration)s\n' \
            '#SBATCH --requeue\n' \
            '#SBATCH --partition=%(queue)s\n' \
            '#SBATCH --output=%(std)s/{id}.o\n' \
            '#SBATCH --error=%(std)s/{id}.e\n' % SYS + dev_flag

        #            '#SBATCH --mail-type=ALL\n' \
        #            '#SBATCH --mail-user=%(email)s\n' \
        def _run_grid(message):
            submit_message = CMD(message)
            lines = submit_message.split('\n')
            submit_message = lines[len(lines) - 1]
            return _line_splitter.split(submit_message)[3]

        self._my_run = DRY_RUN if _args.dry else _run_grid
        with open(SYS.task_dir + '/.tmp', "a") as f:
            print('[task_dir] %s' % SYS.task_dir, file=f)

    def _valid(self, command, id):
        if id in SYS.skip:
            return False
        ret = CMD("squeue -u $(whoami) -o %j", log=False)
        jid = '%(task_name)s:{id}'.format(id=id) % SYS
        return jid not in ret.split('\n')

    def _run(self, id):
        if not _args.dry:
            out_f = '%(std)s/{id}.o'.format(id=id) % SYS
            if os.path.exists(out_f):
                os.remove(out_f)
        self.job_id = self._my_run('sbatch {script}'.format(script=self.script))

    def _after(self, id):
        with open(SYS.task_dir + '/.tmp', "a") as f:
            job_dict = {'job_name': id,
                        'job_id': self.job_id,
                        'ctask_list': self._local_ctask_queue,
                        'rtask_list': self._local_rtask_queue
                        }
            print(json.dumps(job_dict), file=f)

    def _finish(self):
        tmpfile = SYS.task_dir + '/.tmp'
        if _args.dry:
            os.remove(tmpfile)
            return
        info_file = SYS.task_dir + '/info-' + SYS.time + '.json'
        if os.path.isfile(tmpfile):
            if len(self._global_ctask_queue):
                check_finish_cmd = 'python2 ~/tools/marcc_manager.py --task check_succ_loop --input %s' % info_file
                self._local_rtask_queue = [([(check_finish_cmd, 0)], '.succ-%(time)s' % SYS)]
                self._submit_queue('.succ-%(time)s' % SYS)
                self._job_counter -= 1
                sub_cmd = 'screen -d -m -S {id}-check_finish;' \
                          'screen -S {id} -X stuff \'' \
                          '{command};exit' \
                          '\n\''.format(id=SYS.task_name + '-' + SYS.time,
                                        command=check_finish_cmd)
                CMD(sub_cmd)
            os.rename(tmpfile, info_file)
            if int(SYS.get('requeue', 0)) and not _args.dry:
                requeue_cmd = 'python2 ~/tools/marcc_manager.py --task requeue_loop --auto --input %s' % info_file
                sub_cmd = 'screen -d -m -S {id};' \
                          'screen -S {id} -X stuff \'' \
                          '{command};exit' \
                          '\n\''.format(id=SYS.task_name + '-' + SYS.time,
                                        command=requeue_cmd)
                CMD(sub_cmd)


class MarccInteractJobHandler(MarccJobHandler):
    def __init__(self):
        super(MarccInteractJobHandler, self).__init__()
        self.runner = 'srun'
        self._my_run = DRY_RUN if _args.dry else CMD

    def _run(self, id):
        self._my_run('{runner} {script}'.format(runner=self.runner,
                                                script=self.script))


def _git():
    CMD('git pull;git commit -am "SYNC:{}";git push'.format(SYS.set('msg', 'minor')))
    CMD('git describe --always > src/version.txt')


def _upload():
    for rmt in SYS.remote.split(';'):
        CMD('rsync -av *.py src tools %s' % rmt)


def _download():
    for rmt in SYS.remote.split(';'):
        pass


@shepherd(before=[setup])
def git():
    _git()


@shepherd(before=[setup])
def sync_only():
    _download()
    _upload()


@shepherd(before=[setup])
def sync():
    _download()
    _git()
    _upload()


if __name__ == "__main__":
    import imp

    task = _args.task.split('-')[0]
    if task in _collect_exp_functions(EXP_FILE):
        imp.load_source('modulename', EXP_FILE).__dict__[task]()
    elif task in locals():
        locals()[task]()
    else:
        assert False, 'No such experiment: %s' % task
