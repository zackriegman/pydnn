#! /usr/bin/env python

__author__ = 'isaac'

'''
See "~/.boto" for aws_access_key_id and aws_secret_access_key
#g2.2xlarge has NVIDIA Kepler GK104GL [GRID K520] GPU
# CUDA Computer Capability: 3.0

See http://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_DescribeImages.html
for list of valid filters for get_all_images()
'''

import boto.ec2
import boto.ec2.blockdevicemapping
from datetime import datetime
from datetime import timedelta
from time import sleep
import argparse
import subprocess
import shlex
from tools import Dot
import tools

config = tools.load_config('AWS_UTIL_CONFIG', __file__, 'aws_util.conf')


DEFAULT_NAME = config['ec2']['default_name']
REGION = config['ec2']['region']
PEM_NAME = config['ec2']['pem_name']
MAX_PRICE = config['ec2']['max_price']

FROM = config['email']['from']
TO = config['email']['to']
USERNAME = config['email']['username']
PASSWORD = config['email']['password']
SMTP = config['email']['smtp']

conn = boto.ec2.connect_to_region(REGION)

if conn is None:
    raise Exception('connection failed')


def get_image_id_by_name(name):
    image = get_image_by_name(name)
    if image is not None:
        return image.id
    else:
        raise Exception("image '{}' not found".format(name))


def get_recent_gpu_price():
    def get_price_for_zone(zone):
        recent_prices = conn.get_spot_price_history(
            start_time=(datetime.utcnow()-timedelta(seconds=1)).isoformat(),
            end_time=datetime.utcnow().isoformat(),
            instance_type='g2.2xlarge',
            product_description='Linux/UNIX',
            availability_zone=zone
        )
        return recent_prices[0].price

    zones = ['us-east-1' + l for l in ('a', 'b', 'd', 'e')]
    prices = [get_price_for_zone(z) for z in zones]
    print(zip(zones, prices))
    return max(prices), min(prices), zones[prices.index(min(prices))]


def get_running_instances(name=None):
    filters = {'instance-state-name': 'running'}
    if name:
        filters["tag:name"] = name
    return conn.get_only_instances(filters=filters)


def get_unique_instance(name):
    instances = get_running_instances(name)
    if len(instances) > 1:
        raise Exception('more than one instance fits criteria')
    elif len(instances) == 0:
        raise Exception('no instances fit criteria')
    else:
        return instances[0]


def start_spot_instance(name, image_id=None):
    if len(get_running_instances(name)) != 0:
        raise Exception("there is already an instance with that name")

    highest_price, lowest_price, lowest_cost_zone = get_recent_gpu_price()
    print('highest price: {}'.format(highest_price))
    # bid_price = highest_price + 0.03
    if MAX_PRICE < lowest_price * 1.5:
        raise Exception("bid price is less than 1.5 times current prices... too risky")

    print('bidding {} max price to instantiate image "{}" in zone "{}"'.format(
        MAX_PRICE, image_id, lowest_cost_zone))

    sda1 = boto.ec2.blockdevicemapping.EBSBlockDeviceType()
    sda1.size = 10
    bdm = boto.ec2.blockdevicemapping.BlockDeviceMapping()
    bdm['/dev/sda1'] = sda1

    spot_request = conn.request_spot_instances(
        price=str(MAX_PRICE),
        # image_id='ami-9eaa1cf6',  # plain ubuntu
        # image_id='ami-2ca87b44',  # GPU and theano preinstalled
        image_id=image_id,
        count=1,
        type='one-time',
        valid_from=(datetime.utcnow()+timedelta(seconds=5)).isoformat(),
        valid_until=(datetime.utcnow()+timedelta(minutes=15)).isoformat(),
        key_name=PEM_NAME,
        security_groups=['all'],
        instance_type='g2.2xlarge',
        # instance_type='t2.micro',
        # block_device_map=bdm,
        ebs_optimized=True,
        placement=lowest_cost_zone
    )[0]

    print('waiting for spot instance request fulfillment')
    instance_id = None
    dotter = Dot()
    while instance_id is None:
        dotter.dot()
        sleep(10)
        request = conn.get_all_spot_instance_requests(spot_request.id)[0]
        instance_id = request.instance_id
        if instance_id is None and request.state != 'open':
            raise Exception("request {}, status: {}".format(
                request.state, request.status))
    print("\nspot instance fulfilled with id:" + str(instance_id))

    sleep(10)  # give AWS a chance to setup the IP and everything
    instance = conn.get_only_instances(instance_ids=[instance_id])[0]
    instance.add_tag("name", name)
    # login_exec = "ssh -i {} -o StrictHostKeyChecking=no ubuntu@{}".\
    #     format('"' + PEM_FILE + '"', instance.ip_address)
    login_exec = get_login_exec(instance.ip_address)
    sftp_exec = get_sftp_exec(instance.ip_address)
    connect_message = ('connect to instance with "' + login_exec + '" or "' +
                       sftp_exec + '"')
    tools.send_email(
        from_addr=FROM,
        to_addr=TO,
        username=USERNAME,
        password=PASSWORD,
        smtp=SMTP,
        subject='Subject: EC2 spot instance "{}" is ready'.format(name),
        body=connect_message
    )
    print(connect_message)
    return instance_id, login_exec, sftp_exec


def get_login_exec(instance_ip):
    return "ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=100 ubuntu@{}".format(instance_ip)


def get_sftp_exec(instance_ip):
    return "nautilus ssh://ubuntu@{}/home/ubuntu".format(instance_ip)


def get_image_by_name(name):
    images = conn.get_all_images(owners=['self'], filters={'name': name})
    if len(images) > 1:
        raise Exception("more than one image with name {}".format(name))
    if len(images) == 0:
        return None
    else:
        return images[0]


def save_spot_instance(name, image_name, terminate=True):
    instance = get_unique_instance(name)

    image = get_image_by_name(image_name)
    if image is not None:
        print('deleting old persist image')
        image.deregister()

    print('creating new persist image...')
    dotter = Dot()
    image_id = instance.create_image(
        name=image_name,
        description='image used to persist instance',
        no_reboot=False
    )

    sleep(5)  # give AWS a change to realize update its image index

    image = conn.get_all_images(image_ids=[image_id])[0]
    while image.state == 'pending':
        dotter.dot()
        sleep(5)
        image.update()
    if image.state == 'available':
        print('successfully created image')
    else:
        raise Exception(
            "failed to create image with message '{}'".format(image.state))

    print('creating backup of new persist image...')
    conn.copy_image(
        source_region=REGION,
        source_image_id=image_id,
        name="_" + image_name+'_'+datetime.now().strftime('%Y%m%d%H%M%S%f'),
        description='backup'
    )

    if terminate:
        stop_spot_instance_without_saving(instance.id)


def launch_instance(name, image_id):
    instance_id, login_exec, sftp_exec = start_spot_instance(name, image_id)
    sleep(10)  # give the computer a chance to boot up and start SSH server
    subprocess.call(shlex.split(login_exec))


def ssh(name):
    login_exec = get_login_exec(get_unique_instance(name).ip_address)
    print(login_exec)
    subprocess.call(shlex.split(login_exec))


def sftp(name):
    sftp_exec = get_sftp_exec(get_unique_instance(name).ip_address)
    print(sftp_exec)
    subprocess.Popen(shlex.split(sftp_exec))


def stop_all_spot_instances_without_saving():
    instances = get_running_instances()
    for instance in instances:
        if instance.state == 'running' and instance.spot_instance_request_id:
            print("stopping spot instance {}".format(instance.id))
            stop_spot_instance_without_saving(instance.id)
    list_instances()


def print_instances(instances):
    print("-- Instances --")
    for instance in instances:
        if 'name' in instance.tags:
            name = instance.tags['name']
        else:
            name = '<none>'

        print("name: {}, id: {}, state: {}, ip: {}, type: {}, launch time: {}".format(
            name, instance.id, instance.state, instance.ip_address, instance.instance_type,
            instance.launch_time))


def stop_spot_instance_without_saving(instance_id):
    print("terminating instance...")
    conn.terminate_instances([instance_id])


def list_instances():
    instances = get_running_instances()
    print_instances(instances)


def list_amis():
    print('-- AMIs --')
    for image in conn.get_all_images(owners=['self']):
        if image.name[0] != '_':
            print('name: {}, id: {}'.format(image.name, image.id))


def create_security_group():
    print(conn.get_all_security_groups())
    security_group = conn.create_security_group(
        'all',
        'wide open security group'
    )
    security_group.authorize('tcp', 0, 65535, '0.0.0.0/0')
    print(conn.get_all_security_groups())
# create_security_group()


def handle_command_line():
    parser = argparse.ArgumentParser(
        description="start and stop persistent GPU spot instances on Amazon EC2")

    name_parser = argparse.ArgumentParser(add_help=False)
    name_parser.add_argument('instance',
                             nargs='?',
                             type=str,
                             help='name of instance')

    ami_parser = argparse.ArgumentParser(add_help=False)
    ami_parser.add_argument('AMI',
                            nargs='?',
                            type=str,
                            help='name of AMI')

    ami_id_parser = argparse.ArgumentParser(add_help=False)
    ami_id_parser.add_argument('-a', '--AMI_ID', type=str, help='ID of AMI')

    default_msg = ' (using default instance name and AMI name if not provided)'
    default_name_msg = ' (using default instance name if not provided)'

    subparsers = parser.add_subparsers(dest='sub_command')

    def add_cmd(name, desc, func, parents=None):
        if parents is None:
            option = subparsers.add_parser(name,
                                           help=desc,
                                           description=desc)
        else:
            option = subparsers.add_parser(name,
                                           help=desc,
                                           description=desc,
                                           parents=parents)
        option.set_defaults(func=func)
        return option

    def default_start_ami_id(arg):
        if arg.AMI_ID is not None:
            if arg.AMI is not None:
                raise Exception('only specify one of AMI name or AMI id')
            name = conn.get_all_images(image_ids=[arg.AMI_ID])[0].name
            return arg.AMI_ID, name
        else:
            if arg.AMI is not None:
                return get_image_id_by_name(arg.AMI), arg.AMI
            else:
                amis = {image.name: image for image in
                        conn.get_all_images(owners=['self'])
                        if image.name[0] != '_'}
                if len(amis) == 1:
                    return amis.values()[0].id, amis.values()[0]
                elif DEFAULT_NAME in amis:
                    return amis[DEFAULT_NAME].id, DEFAULT_NAME
                elif len(amis) == 0:
                    raise Exception('no AMIs available to load')
                else:
                    raise Exception('please specify AMI to load ' + str(amis))

    def default_start_instance(arg, ami_name):
        print("default_start_instance", arg, ami_name)
        if arg.instance:
            return arg.instance
        else:
            instances = [instance.tags['name'] for instance in
                         get_running_instances()]
            print("default_start_instance:instances", instances)
            if ami_name not in instances:
                return ami_name
            else:
                suffix = 0
                while (ami_name + '_' + str(suffix)) in instances:
                    suffix += 1
                return ami_name + '_' + str(suffix)

    def default_save_ami_name(arg, instance_name):
        if arg.AMI is not None:
            return arg.AMI
        else:
            return instance_name

    def default_running_instance(arg):
        if arg.instance:
            return arg.instance
        else:
            instances = [instance.tags['name'] for instance in
                         get_running_instances()]
            if len(instances) == 1:
                return instances[0]
            elif DEFAULT_NAME in instances:
                return DEFAULT_NAME
            elif len(instances) == 0:
                raise Exception('no instances are running')
            else:
                raise Exception('please specify an instance ' + str(instances))

    ################
    # sub commands #
    ################

    def cmd_start(arg):
        ami_id, ami_name = default_start_ami_id(arg)
        instance_name = default_start_instance(arg, ami_name)
        print('starting "{}" as "{}"'.format(ami_name, instance_name))
        start_spot_instance(name=instance_name, image_id=ami_id)
    add_cmd('start', 'start a spot instance' + default_msg,
            cmd_start, [ami_parser, name_parser, ami_id_parser])

    def cmd_launch(arg):
        ami_id, ami_name = default_start_ami_id(arg)
        instance_name = default_start_instance(arg, ami_name)
        print('starting "{}" as "{}"'.format(ami_name, instance_name))
        launch_instance(name=instance_name, image_id=ami_id)
    add_cmd('launch', 'start a spot instance and log in via SSH' + default_msg,
            cmd_launch, [ami_parser, name_parser, ami_id_parser])

    def cmd_ssh(arg):
        name = default_running_instance(arg)
        print('logging into "{}"'.format(name))
        ssh(name=name)
    add_cmd('ssh', 'log into spot instance' + default_name_msg,
            cmd_ssh, [name_parser])

    def cmd_sftp(arg):
        name = default_running_instance(arg)
        print('opening sftp connection to "{}"'.format(name))
        sftp(name=name)
    add_cmd('sftp', 'open sftp connection in file browser' + default_name_msg,
            cmd_sftp, [name_parser])

    def cmd_save(arg):
        instance_name = default_running_instance(arg)
        ami_name = default_save_ami_name(arg, instance_name)
        print('saving "{}" as "{}"'.format(instance_name, ami_name))
        save_spot_instance(name=instance_name, image_name=ami_name, terminate=False)
    add_cmd('save', 'save spot instance and reboot' + default_msg,
            cmd_save, [name_parser, ami_parser])

    def cmd_persist(arg):
        instance_name = default_running_instance(arg)
        ami_name = default_save_ami_name(arg, instance_name)
        print('persisting "{}" as "{}"'.format(instance_name, ami_name))
        save_spot_instance(name=instance_name, image_name=ami_name)
    add_cmd('persist', 'save and terminate spot instance' + default_msg,
            cmd_persist, [name_parser, ami_parser])

    def cmd_kill(arg):
        if arg.all:
            print('killing all instances')
            stop_all_spot_instances_without_saving()
        elif arg.name:
            inst = get_unique_instance(args.name)
            print('killing "{}"'.format(args.name))
            stop_spot_instance_without_saving(inst.id)
        else:
            print('must specify instance to kill')
    kill_parser = add_cmd('kill', 'kill spot instance(s)', cmd_kill)
    kill_group = kill_parser.add_mutually_exclusive_group()
    kill_group.add_argument('name', nargs='?', type=str, help='name of instance to kill')
    kill_group.add_argument('--all', action='store_true')

    def cmd_list(arg):
        if arg.instances:
            list_instances()
        elif arg.amis:
            list_amis()
        else:
            list_instances()
            list_amis()
    list_parser = add_cmd('list', 'list all active instances and AMIs associated with account '
                          '(except AMIs starting with "_")', cmd_list)
    list_group = list_parser.add_mutually_exclusive_group()
    list_group.add_argument('-i', '--instances', action='store_true')
    list_group.add_argument('-a', '--amis', action='store_true')

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    handle_command_line()