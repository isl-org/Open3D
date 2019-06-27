#!/usr/bin/env python

"""Script to import GitHub issues into Asana
    v2.0

    Nestor Subiron (v1): nsubiron@cvc.uab.es
    German Ros (v2): german.ros@intel.com

    Requirements:
        pip install asana
        pip install PyGithub
"""

# pylint: disable=missing-docstring
# pylint: disable=no-member
# pylint: disable=too-few-public-methods


import argparse
from argparse import RawTextHelpFormatter
import json

import asana
import github

class AsanaClient:
    """
    Basic wrapper for the ASANA API. This class can query workspaces, users, projects and tasks.
    It can also create new tasks and has utilities to convert from GitHub issues to ASANA tasks.
    """
    def __init__(self, scheme):
        """

        :param scheme: A dictionary containing the json options to access ASANA as follows:

          {
            "asana_user_token": "XXXXX",
            "workspace": "XXXXX",
            "project": {"TEST": "XXXX"},
            "github_section": {"GITHUB": "XXX"},
            "custom_fields": {"GithubID": "XXXX"}
          },

        """
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
        for sub_project in projects:
            if sub_project['name'] == project:
                project_id = sub_project['id']
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

                task_dict[key] = (value, str(custom_field['id']))

            output_tasks.append(task_dict)

        return output_tasks

    def create_task(self, task_dict):
        self.client.tasks.create(params=task_dict)

    def convert_issue_to_task(self, github_issue_dict, github_to_asana_map):
        # pylint: disable=line-too-long

        """

        :param github_issue_dict: dictionary describing github issue with the following fields:

                 {
                        'id': issue.id,
                        'number': issue.number,
                        'url': issue.url,
                        'title': issue.title,
                        'user': issue.user.login,
                        'state': issue.state,
                        'body': issue.body,
                        'assignees': [list-of-strings],
                        'labels': [list-of-strings],
                    }

        :param github_to_asana_map: dictionary defining a map from github properties to asana properties. E.g.:

                {"number": "GithubID"}

        :return: dictionary containing the fields of an ASANA task:

                {'name': -,
                   'id': string,
                   'gid': string,
                   'completed': string,
                   'projects': [list-of-strings],
                   'memberships': string,
                   'custom_fields': [list-of-strings],
                   'notes': string
                }
        """
        custom_fields = {}
        for key_github, key_asana in github_to_asana_map.items():
            if key_asana in self._scheme['custom_fields']:
                numerical_key_asana = self._scheme['custom_fields'][key_asana]
                custom_fields[numerical_key_asana] = str(github_issue_dict[key_github])

        project_code = list(self._scheme['project'].values())[0]
        project_section = list(self._scheme['github_section'].values())[0]

        str_notes = 'Created by user [{}]\n\n' \
                    'URL: {}\n\n' \
                    'Assigned to [{}]\n\n' \
                    'Body: {}\n\n' \
                    'Labels: {}\n\n'.format(github_issue_dict['user'],
                                            github_issue_dict['url'],
                                            github_issue_dict['assignees'],
                                            github_issue_dict['body'],
                                            github_issue_dict['labels'])

        str_membership = [{'project': project_code, 'section': project_section}]

        asana_task_dict = {'name': github_issue_dict['title'],
                           'id': github_issue_dict['number'],
                           'gid': github_issue_dict['id'],
                           'completed': github_issue_dict['state'] == 'completed',
                           'projects': [project_code],
                           'memberships': str_membership,
                           'custom_fields': custom_fields,
                           'notes': str_notes
                           }

        return asana_task_dict


class GithubClient:
    """
    Basic wrapper to the GitHub API. It is able to retrieve issues from a github repository.
    """
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
                        'user': issue.user.login,
                        'state': issue.state,
                        'body': issue.body,
                        'assignees': [str(x.login) for x in issue.assignees],
                        'labels': [str(x.name) for x in issue.labels],
                    }

                    output_issues.append(issue_dict)

        return output_issues

def parse_json(config):
    with open(config) as file:
        scheme = json.load(file)

    return scheme

def main(args):
    scheme = parse_json(args.config)

    if not 'asana_env' in scheme:
        raise ValueError("[asana_env] variable not found.")

    asana_client = AsanaClient(scheme['asana_env'])
    print('Hello {}!'.format(asana_client.get_username()))

    project  = list(scheme['asana_env']['project'].keys())[0]
    asana_tasks = asana_client.get_tasks(project)
    exclude_ids = []
    for task in asana_tasks:
        if 'GithubID' in task and task['GithubID'][0]:
            exclude_ids.append(task['GithubID'][0])

    # remove duplicates
    exclude_ids = list(set(exclude_ids))

    github_client = GithubClient(scheme['github_env'])
    github_issues = github_client.get_issues(exclude_ids)

    # transform github issues into asana tasks
    for idx, issue in enumerate(github_issues):
        print('* Importing issue [{}/{}]: {}'.format(idx, len(github_issues), issue['title']))
        task = asana_client.convert_issue_to_task(issue, scheme['github_to_asana_map'])
        asana_client.create_task(task)


if __name__ == '__main__':
    DESCRIPTION = ("Script to import GitHub issues into Asana\n")
    PARSER = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=RawTextHelpFormatter)
    PARSER.add_argument('--config', help='Configuration file (json)')

    ARGS = PARSER.parse_args()

    try:
        main(ARGS)
    except KeyboardInterrupt:
        print('\nCancelled.')
