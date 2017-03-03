#ifndef INTERFACE_LUA_
#define INTERFACE_LUA_

#include <msgpack.hpp>
#include <zmq.hpp>

namespace cpprob{

template<class T>
T receive(zmq::socket_t& socket){
    zmq::message_t reply;
    socket.recv (&reply);

    std::string str = std::string(static_cast<char*>(reply.data()), reply.size());
    msgpack::object_handle oh = msgpack::unpack(str.data(), str.size());
    return oh.get().as<T>();
}

template<class T>
void send(T&& obj, zmq::socket_t& socket){
    msgpack::sbuffer sbuf;
    msgpack::pack(sbuf, obj);

    zmq::message_t request (sbuf.size());
    memcpy (request.data(), sbuf.data(), sbuf.size());
    socket.send (request);
}

}
#endif  // INTERFACE_LUA_
