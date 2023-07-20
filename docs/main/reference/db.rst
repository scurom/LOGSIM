Database
====================================================

The database module contains the classes that are used to interact with a MongoDB database.
The `connection string <https://www.mongodb.com/docs/manual/reference/connection-string/>`_ should be saved in the environment variable ``LOGSIM_URI``.
The connection is performed with the package `pymongo <https://pymongo.readthedocs.io/en/stable/>`_.
The database name used is ``DB``.
The DB class is used in the other modules when data should be written to the database, like run results.

.. automodule:: logsim.db
    :members:
    :undoc-members:
    :show-inheritance:
