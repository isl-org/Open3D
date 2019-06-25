#!/usr/bin/env python

"""Script to import GitHub issues into Asana
    v2.0

    Nestor Subiron (v1): nsubiron@cvc.uab.es
    German Ros (v2): german.ros@intel.com

    Requirements:
        pip install asana
        pip install PyGithub
"""

import argparse
from argparse import RawTextHelpFormatter
import json

import asana
import github

import pdb

class AsanaClient(object):
    def __init__(self, scheme):
        self._scheme = scheme
        self.client = asana.Client.basic_auth(self._scheme['asana_user_token'])

    def get_username(self):
        return self.client.users.me()['name']

    def get_users(self):
        return self.client.users.find_by_workspace(self._scheme['workspace'])

    def get_workspaces(self):
        return self.client.users.me()['workspaces']

    def get_projects(self):
        return self.client.projects.find_all({'workspace': self._scheme['workspace']})

    def get_tasks(self, project, only_not_completed=True):
        opts = {"opt_fields": "name, custom_fields, completed"}
        output_tasks = []
        project_id = None

        projects = self.get_projects()
        for p in projects:
            if p['name'] == project:
                project_id = p['id']
                break

        if not project_id:
            return output_tasks

        for task in self.client.tasks.find_by_project(project_id, opts, iterator_type=None):
            if only_not_completed and task['completed']:
                continue

            task_dict = {
                'name': task['name'],
                'id': task['id'],
                'gid': task['gid'],
                'completed': task['completed']
            }

            for custom_field in task['custom_fields']:
                key = custom_field['name']

                if custom_field['type'] == 'text':
                    value = custom_field['text_value']
                elif custom_field['type'] == 'enum':
                    value = custom_field['enum_value']

                task_dict[key] = value

            output_tasks.append(task_dict)

        return output_tasks

class GithubClient(object):
    def __init__(self, scheme):
        self._scheme = scheme
        self.client = github.Github(self._scheme['github_user_token'])

    def get_issues(self, exclude_ids=None):
        output_issues = []

        if exclude_ids:
            hash_ids = {i: True for i in exclude_ids}

        repo = self.client.get_repo(self._scheme['github_repo'])
        labels = [x for x in repo.get_labels()]

        for label in labels:
            for issue in repo.get_issues(state='open', labels=[label]):
                if exclude_ids and str(issue.number) in hash_ids:
                    continue

                if not issue.pull_request:
                    issue_dict = {
                        'id': issue.id,
                        'number': issue.number,
                        'url': issue.url,
                        'title': issue.title,
                        'user': issue.user,
                        'state': issue.state,
                        'body': issue.body,
                        'assignees': [str(x) for x in issue.assignees],
                        'labels': [str(x) for x in issue.labels],
                    }

                    output_issues.append(issue_dict)

        return output_issues

def parse_json(config):
    with open(config) as fd:
        scheme = json.load(fd)

    return scheme

def main(args):
    scheme = parse_json(args.config)

    if not 'asana_env' in scheme:
        raise ValueError("[asana_env] variable not found.")

    asana_client = AsanaClient(scheme['asana_env'])
    print('Hello {}!'.format(asana_client.get_username()))

    asana_tasks = asana_client.get_tasks('TEST')
    exclude_ids = []
    for task in asana_tasks:
        if 'GithubID' in task and task['GithubID']:
            exclude_ids.append(task['GithubID'])

    github_client = GithubClient(scheme['github_env'])
    github_issues = github_client.get_issues(exclude_ids)

    #for issue in issues:
    #    print(issue)

if __name__ == '__main__':
    description = ("Script to import GitHub issues into Asana\n")
    parser = argparse.ArgumentParser(description=description, formatter_class=RawTextHelpFormatter)
    parser.add_argument('--config', help='Configuration file (json)')

    args = parser.parse_args()

    try:
        main(args)
    except KeyboardInterrupt:
        print('\nCancelled.')