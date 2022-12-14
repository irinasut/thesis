# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from proto.scannerservice import scannerservice_pb2 as proto_dot_scannerservice_dot_scannerservice__pb2


class HardwareControllerStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.JumpToPosition = channel.unary_unary(
                '/scannerservice.HardwareController/JumpToPosition',
                request_serializer=proto_dot_scannerservice_dot_scannerservice__pb2.Position.SerializeToString,
                response_deserializer=proto_dot_scannerservice_dot_scannerservice__pb2.Time.FromString,
                )
        self.ScanWorkPlane = channel.unary_unary(
                '/scannerservice.HardwareController/ScanWorkPlane',
                request_serializer=proto_dot_scannerservice_dot_scannerservice__pb2.WorkPlaneMessage.SerializeToString,
                response_deserializer=proto_dot_scannerservice_dot_scannerservice__pb2.Time.FromString,
                )
        self.GetHardScanField = channel.unary_unary(
                '/scannerservice.HardwareController/GetHardScanField',
                request_serializer=proto_dot_scannerservice_dot_scannerservice__pb2.Empty.SerializeToString,
                response_deserializer=proto_dot_scannerservice_dot_scannerservice__pb2.ScanFieldConfig.FromString,
                )


class HardwareControllerServicer(object):
    """Missing associated documentation comment in .proto file."""

    def JumpToPosition(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ScanWorkPlane(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetHardScanField(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_HardwareControllerServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'JumpToPosition': grpc.unary_unary_rpc_method_handler(
                    servicer.JumpToPosition,
                    request_deserializer=proto_dot_scannerservice_dot_scannerservice__pb2.Position.FromString,
                    response_serializer=proto_dot_scannerservice_dot_scannerservice__pb2.Time.SerializeToString,
            ),
            'ScanWorkPlane': grpc.unary_unary_rpc_method_handler(
                    servicer.ScanWorkPlane,
                    request_deserializer=proto_dot_scannerservice_dot_scannerservice__pb2.WorkPlaneMessage.FromString,
                    response_serializer=proto_dot_scannerservice_dot_scannerservice__pb2.Time.SerializeToString,
            ),
            'GetHardScanField': grpc.unary_unary_rpc_method_handler(
                    servicer.GetHardScanField,
                    request_deserializer=proto_dot_scannerservice_dot_scannerservice__pb2.Empty.FromString,
                    response_serializer=proto_dot_scannerservice_dot_scannerservice__pb2.ScanFieldConfig.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'scannerservice.HardwareController', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class HardwareController(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def JumpToPosition(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/scannerservice.HardwareController/JumpToPosition',
            proto_dot_scannerservice_dot_scannerservice__pb2.Position.SerializeToString,
            proto_dot_scannerservice_dot_scannerservice__pb2.Time.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ScanWorkPlane(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/scannerservice.HardwareController/ScanWorkPlane',
            proto_dot_scannerservice_dot_scannerservice__pb2.WorkPlaneMessage.SerializeToString,
            proto_dot_scannerservice_dot_scannerservice__pb2.Time.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetHardScanField(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/scannerservice.HardwareController/GetHardScanField',
            proto_dot_scannerservice_dot_scannerservice__pb2.Empty.SerializeToString,
            proto_dot_scannerservice_dot_scannerservice__pb2.ScanFieldConfig.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)


class ServiceControllerStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.JumpToPosition = channel.unary_unary(
                '/scannerservice.ServiceController/JumpToPosition',
                request_serializer=proto_dot_scannerservice_dot_scannerservice__pb2.Position.SerializeToString,
                response_deserializer=proto_dot_scannerservice_dot_scannerservice__pb2.Time.FromString,
                )
        self.ScanWorkPlane = channel.unary_unary(
                '/scannerservice.ServiceController/ScanWorkPlane',
                request_serializer=proto_dot_scannerservice_dot_scannerservice__pb2.WorkPlaneMessage.SerializeToString,
                response_deserializer=proto_dot_scannerservice_dot_scannerservice__pb2.ProtoPointer.FromString,
                )
        self.RescanWorkPlane = channel.unary_unary(
                '/scannerservice.ServiceController/RescanWorkPlane',
                request_serializer=proto_dot_scannerservice_dot_scannerservice__pb2.ProtoPointer.SerializeToString,
                response_deserializer=proto_dot_scannerservice_dot_scannerservice__pb2.Time.FromString,
                )
        self.GetScannedWorkPlanes = channel.unary_unary(
                '/scannerservice.ServiceController/GetScannedWorkPlanes',
                request_serializer=proto_dot_scannerservice_dot_scannerservice__pb2.WorkPlaneMessage.SerializeToString,
                response_deserializer=proto_dot_scannerservice_dot_scannerservice__pb2.Time.FromString,
                )
        self.GetScannedWorkPlane = channel.unary_unary(
                '/scannerservice.ServiceController/GetScannedWorkPlane',
                request_serializer=proto_dot_scannerservice_dot_scannerservice__pb2.ProtoPointer.SerializeToString,
                response_deserializer=proto_dot_scannerservice_dot_scannerservice__pb2.WorkPlaneMessage.FromString,
                )


class ServiceControllerServicer(object):
    """Missing associated documentation comment in .proto file."""

    def JumpToPosition(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ScanWorkPlane(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def RescanWorkPlane(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetScannedWorkPlanes(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetScannedWorkPlane(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ServiceControllerServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'JumpToPosition': grpc.unary_unary_rpc_method_handler(
                    servicer.JumpToPosition,
                    request_deserializer=proto_dot_scannerservice_dot_scannerservice__pb2.Position.FromString,
                    response_serializer=proto_dot_scannerservice_dot_scannerservice__pb2.Time.SerializeToString,
            ),
            'ScanWorkPlane': grpc.unary_unary_rpc_method_handler(
                    servicer.ScanWorkPlane,
                    request_deserializer=proto_dot_scannerservice_dot_scannerservice__pb2.WorkPlaneMessage.FromString,
                    response_serializer=proto_dot_scannerservice_dot_scannerservice__pb2.ProtoPointer.SerializeToString,
            ),
            'RescanWorkPlane': grpc.unary_unary_rpc_method_handler(
                    servicer.RescanWorkPlane,
                    request_deserializer=proto_dot_scannerservice_dot_scannerservice__pb2.ProtoPointer.FromString,
                    response_serializer=proto_dot_scannerservice_dot_scannerservice__pb2.Time.SerializeToString,
            ),
            'GetScannedWorkPlanes': grpc.unary_unary_rpc_method_handler(
                    servicer.GetScannedWorkPlanes,
                    request_deserializer=proto_dot_scannerservice_dot_scannerservice__pb2.WorkPlaneMessage.FromString,
                    response_serializer=proto_dot_scannerservice_dot_scannerservice__pb2.Time.SerializeToString,
            ),
            'GetScannedWorkPlane': grpc.unary_unary_rpc_method_handler(
                    servicer.GetScannedWorkPlane,
                    request_deserializer=proto_dot_scannerservice_dot_scannerservice__pb2.ProtoPointer.FromString,
                    response_serializer=proto_dot_scannerservice_dot_scannerservice__pb2.WorkPlaneMessage.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'scannerservice.ServiceController', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class ServiceController(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def JumpToPosition(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/scannerservice.ServiceController/JumpToPosition',
            proto_dot_scannerservice_dot_scannerservice__pb2.Position.SerializeToString,
            proto_dot_scannerservice_dot_scannerservice__pb2.Time.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ScanWorkPlane(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/scannerservice.ServiceController/ScanWorkPlane',
            proto_dot_scannerservice_dot_scannerservice__pb2.WorkPlaneMessage.SerializeToString,
            proto_dot_scannerservice_dot_scannerservice__pb2.ProtoPointer.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def RescanWorkPlane(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/scannerservice.ServiceController/RescanWorkPlane',
            proto_dot_scannerservice_dot_scannerservice__pb2.ProtoPointer.SerializeToString,
            proto_dot_scannerservice_dot_scannerservice__pb2.Time.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetScannedWorkPlanes(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/scannerservice.ServiceController/GetScannedWorkPlanes',
            proto_dot_scannerservice_dot_scannerservice__pb2.WorkPlaneMessage.SerializeToString,
            proto_dot_scannerservice_dot_scannerservice__pb2.Time.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetScannedWorkPlane(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/scannerservice.ServiceController/GetScannedWorkPlane',
            proto_dot_scannerservice_dot_scannerservice__pb2.ProtoPointer.SerializeToString,
            proto_dot_scannerservice_dot_scannerservice__pb2.WorkPlaneMessage.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
