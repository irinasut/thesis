# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2
from proto.axesservice.linear.v2 import linear_axesservice_pb2 as proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2


class HardwareControllerStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetPosition = channel.unary_unary(
                '/axesservice.linear.v2.HardwareController/GetPosition',
                request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
                response_deserializer=proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearPosition.FromString,
                )
        self.GetPositionLimits = channel.unary_unary(
                '/axesservice.linear.v2.HardwareController/GetPositionLimits',
                request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
                response_deserializer=proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearPositionLimit.FromString,
                )
        self.MoveToPosition = channel.unary_unary(
                '/axesservice.linear.v2.HardwareController/MoveToPosition',
                request_serializer=proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearMoveRequest.SerializeToString,
                response_deserializer=proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearMoveResponse.FromString,
                )


class HardwareControllerServicer(object):
    """Missing associated documentation comment in .proto file."""

    def GetPosition(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetPositionLimits(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def MoveToPosition(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_HardwareControllerServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'GetPosition': grpc.unary_unary_rpc_method_handler(
                    servicer.GetPosition,
                    request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                    response_serializer=proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearPosition.SerializeToString,
            ),
            'GetPositionLimits': grpc.unary_unary_rpc_method_handler(
                    servicer.GetPositionLimits,
                    request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                    response_serializer=proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearPositionLimit.SerializeToString,
            ),
            'MoveToPosition': grpc.unary_unary_rpc_method_handler(
                    servicer.MoveToPosition,
                    request_deserializer=proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearMoveRequest.FromString,
                    response_serializer=proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearMoveResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'axesservice.linear.v2.HardwareController', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class HardwareController(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def GetPosition(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/axesservice.linear.v2.HardwareController/GetPosition',
            google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearPosition.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetPositionLimits(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/axesservice.linear.v2.HardwareController/GetPositionLimits',
            google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearPositionLimit.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def MoveToPosition(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/axesservice.linear.v2.HardwareController/MoveToPosition',
            proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearMoveRequest.SerializeToString,
            proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearMoveResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)


class MoveWithScalarVelocityPluginStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetScalarVelocityLimit = channel.unary_unary(
                '/axesservice.linear.v2.MoveWithScalarVelocityPlugin/GetScalarVelocityLimit',
                request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
                response_deserializer=proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearScalarVelocityLimit.FromString,
                )
        self.MoveToPositionWithScalarVelocity = channel.unary_unary(
                '/axesservice.linear.v2.MoveWithScalarVelocityPlugin/MoveToPositionWithScalarVelocity',
                request_serializer=proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearMoveRequestWithScalarVelocity.SerializeToString,
                response_deserializer=proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearMoveResponse.FromString,
                )


class MoveWithScalarVelocityPluginServicer(object):
    """Missing associated documentation comment in .proto file."""

    def GetScalarVelocityLimit(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def MoveToPositionWithScalarVelocity(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_MoveWithScalarVelocityPluginServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'GetScalarVelocityLimit': grpc.unary_unary_rpc_method_handler(
                    servicer.GetScalarVelocityLimit,
                    request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                    response_serializer=proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearScalarVelocityLimit.SerializeToString,
            ),
            'MoveToPositionWithScalarVelocity': grpc.unary_unary_rpc_method_handler(
                    servicer.MoveToPositionWithScalarVelocity,
                    request_deserializer=proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearMoveRequestWithScalarVelocity.FromString,
                    response_serializer=proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearMoveResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'axesservice.linear.v2.MoveWithScalarVelocityPlugin', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class MoveWithScalarVelocityPlugin(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def GetScalarVelocityLimit(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/axesservice.linear.v2.MoveWithScalarVelocityPlugin/GetScalarVelocityLimit',
            google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearScalarVelocityLimit.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def MoveToPositionWithScalarVelocity(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/axesservice.linear.v2.MoveWithScalarVelocityPlugin/MoveToPositionWithScalarVelocity',
            proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearMoveRequestWithScalarVelocity.SerializeToString,
            proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearMoveResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)


class MoveWithVectorVelocityPluginStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetVectorVelocityLimit = channel.unary_unary(
                '/axesservice.linear.v2.MoveWithVectorVelocityPlugin/GetVectorVelocityLimit',
                request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
                response_deserializer=proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearVectorVelocityLimit.FromString,
                )
        self.MoveToPositionWithVectorVelocity = channel.unary_unary(
                '/axesservice.linear.v2.MoveWithVectorVelocityPlugin/MoveToPositionWithVectorVelocity',
                request_serializer=proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearMoveRequestWithVectorVelocity.SerializeToString,
                response_deserializer=proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearMoveResponse.FromString,
                )


class MoveWithVectorVelocityPluginServicer(object):
    """Missing associated documentation comment in .proto file."""

    def GetVectorVelocityLimit(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def MoveToPositionWithVectorVelocity(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_MoveWithVectorVelocityPluginServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'GetVectorVelocityLimit': grpc.unary_unary_rpc_method_handler(
                    servicer.GetVectorVelocityLimit,
                    request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                    response_serializer=proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearVectorVelocityLimit.SerializeToString,
            ),
            'MoveToPositionWithVectorVelocity': grpc.unary_unary_rpc_method_handler(
                    servicer.MoveToPositionWithVectorVelocity,
                    request_deserializer=proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearMoveRequestWithVectorVelocity.FromString,
                    response_serializer=proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearMoveResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'axesservice.linear.v2.MoveWithVectorVelocityPlugin', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class MoveWithVectorVelocityPlugin(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def GetVectorVelocityLimit(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/axesservice.linear.v2.MoveWithVectorVelocityPlugin/GetVectorVelocityLimit',
            google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearVectorVelocityLimit.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def MoveToPositionWithVectorVelocity(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/axesservice.linear.v2.MoveWithVectorVelocityPlugin/MoveToPositionWithVectorVelocity',
            proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearMoveRequestWithVectorVelocity.SerializeToString,
            proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearMoveResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)


class MoveWithScalarAccelerationPluginStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetScalarAccelerationLimit = channel.unary_unary(
                '/axesservice.linear.v2.MoveWithScalarAccelerationPlugin/GetScalarAccelerationLimit',
                request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
                response_deserializer=proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearScalarAccelerationLimit.FromString,
                )
        self.MoveToPositionWithScalarAcceleration = channel.unary_unary(
                '/axesservice.linear.v2.MoveWithScalarAccelerationPlugin/MoveToPositionWithScalarAcceleration',
                request_serializer=proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearMoveRequestWithScalarAcceleration.SerializeToString,
                response_deserializer=proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearMoveResponse.FromString,
                )


class MoveWithScalarAccelerationPluginServicer(object):
    """Missing associated documentation comment in .proto file."""

    def GetScalarAccelerationLimit(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def MoveToPositionWithScalarAcceleration(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_MoveWithScalarAccelerationPluginServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'GetScalarAccelerationLimit': grpc.unary_unary_rpc_method_handler(
                    servicer.GetScalarAccelerationLimit,
                    request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                    response_serializer=proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearScalarAccelerationLimit.SerializeToString,
            ),
            'MoveToPositionWithScalarAcceleration': grpc.unary_unary_rpc_method_handler(
                    servicer.MoveToPositionWithScalarAcceleration,
                    request_deserializer=proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearMoveRequestWithScalarAcceleration.FromString,
                    response_serializer=proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearMoveResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'axesservice.linear.v2.MoveWithScalarAccelerationPlugin', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class MoveWithScalarAccelerationPlugin(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def GetScalarAccelerationLimit(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/axesservice.linear.v2.MoveWithScalarAccelerationPlugin/GetScalarAccelerationLimit',
            google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearScalarAccelerationLimit.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def MoveToPositionWithScalarAcceleration(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/axesservice.linear.v2.MoveWithScalarAccelerationPlugin/MoveToPositionWithScalarAcceleration',
            proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearMoveRequestWithScalarAcceleration.SerializeToString,
            proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearMoveResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)


class MoveWithVectorAccelerationPluginStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetVectorAccelerationLimit = channel.unary_unary(
                '/axesservice.linear.v2.MoveWithVectorAccelerationPlugin/GetVectorAccelerationLimit',
                request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
                response_deserializer=proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearVectorAccelerationLimit.FromString,
                )
        self.MoveToPositionWithVectorAcceleration = channel.unary_unary(
                '/axesservice.linear.v2.MoveWithVectorAccelerationPlugin/MoveToPositionWithVectorAcceleration',
                request_serializer=proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearMoveRequestWithVectorAcceleration.SerializeToString,
                response_deserializer=proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearMoveResponse.FromString,
                )


class MoveWithVectorAccelerationPluginServicer(object):
    """Missing associated documentation comment in .proto file."""

    def GetVectorAccelerationLimit(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def MoveToPositionWithVectorAcceleration(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_MoveWithVectorAccelerationPluginServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'GetVectorAccelerationLimit': grpc.unary_unary_rpc_method_handler(
                    servicer.GetVectorAccelerationLimit,
                    request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                    response_serializer=proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearVectorAccelerationLimit.SerializeToString,
            ),
            'MoveToPositionWithVectorAcceleration': grpc.unary_unary_rpc_method_handler(
                    servicer.MoveToPositionWithVectorAcceleration,
                    request_deserializer=proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearMoveRequestWithVectorAcceleration.FromString,
                    response_serializer=proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearMoveResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'axesservice.linear.v2.MoveWithVectorAccelerationPlugin', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class MoveWithVectorAccelerationPlugin(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def GetVectorAccelerationLimit(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/axesservice.linear.v2.MoveWithVectorAccelerationPlugin/GetVectorAccelerationLimit',
            google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearVectorAccelerationLimit.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def MoveToPositionWithVectorAcceleration(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/axesservice.linear.v2.MoveWithVectorAccelerationPlugin/MoveToPositionWithVectorAcceleration',
            proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearMoveRequestWithVectorAcceleration.SerializeToString,
            proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearMoveResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)


class MoveWithTriggerPluginStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetTriggerStepLimit = channel.unary_unary(
                '/axesservice.linear.v2.MoveWithTriggerPlugin/GetTriggerStepLimit',
                request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
                response_deserializer=proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearTriggerStepLimit.FromString,
                )
        self.MoveToPositionWithTrigger = channel.unary_unary(
                '/axesservice.linear.v2.MoveWithTriggerPlugin/MoveToPositionWithTrigger',
                request_serializer=proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearMoveRequestWithTrigger.SerializeToString,
                response_deserializer=proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearMoveResponse.FromString,
                )


class MoveWithTriggerPluginServicer(object):
    """Missing associated documentation comment in .proto file."""

    def GetTriggerStepLimit(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def MoveToPositionWithTrigger(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_MoveWithTriggerPluginServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'GetTriggerStepLimit': grpc.unary_unary_rpc_method_handler(
                    servicer.GetTriggerStepLimit,
                    request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                    response_serializer=proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearTriggerStepLimit.SerializeToString,
            ),
            'MoveToPositionWithTrigger': grpc.unary_unary_rpc_method_handler(
                    servicer.MoveToPositionWithTrigger,
                    request_deserializer=proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearMoveRequestWithTrigger.FromString,
                    response_serializer=proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearMoveResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'axesservice.linear.v2.MoveWithTriggerPlugin', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class MoveWithTriggerPlugin(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def GetTriggerStepLimit(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/axesservice.linear.v2.MoveWithTriggerPlugin/GetTriggerStepLimit',
            google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearTriggerStepLimit.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def MoveToPositionWithTrigger(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/axesservice.linear.v2.MoveWithTriggerPlugin/MoveToPositionWithTrigger',
            proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearMoveRequestWithTrigger.SerializeToString,
            proto_dot_axesservice_dot_linear_dot_v2_dot_linear__axesservice__pb2.LinearMoveResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)